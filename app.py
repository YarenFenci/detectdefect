import re
from collections import deque
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Config
# ----------------------------
CANDIDATE_COS_THRESHOLD = 0.55        # TF-IDF pre-filter (broader net)
TOP_K_NEIGHBORS = 50

SAFE_TOKEN_MIN = 15                    # relaxed — semantic compensates

# Strict SAFE thresholds (token-level)
SAFE_JACCARD_THRESHOLD = 0.85
SAFE_MIN_SHARED_TOKENS = 8
SAFE_MIN_SEMANTIC_COSINE = 0.78

# Semantic-only SAFE (when sentence-transformers kicks in)
SEMANTIC_EMB_SAFE_THRESHOLD = 0.82    # embedding cosine → SAFEDELETESTRICT
SEMANTIC_EMB_REVIEW_THRESHOLD = 0.60  # embedding cosine → QA_REVIEW

# User-flow / scenario keywords grouped by flow
FLOW_GROUPS: Dict[str, Set[str]] = {
    "auth":        {"login","logout","signin","signup","register","otp","verification","verify",
                    "password","pin","biometrics","fingerprint","faceid","2fa","authentication"},
    "messaging":   {"message","chat","send","receive","delivery","delivered","read","unread",
                    "typing","sticker","emoji","gif","media","photo","video","file","document",
                    "forward","reply","delete","unsend","attachment"},
    "calling":     {"call","voice","videocall","video","ringing","ring","answer","decline",
                    "reject","missed","mute","speaker","bluetooth","headset","mic","microphone",
                    "call_quality","echo","noise"},
    "notification":{"notification","push","badge","sound","vibration","alert","banner"},
    "channel":     {"channel","discovery","explore","search","find","broadcast"},
    "story":       {"story","status","highlight","viewer","reaction"},
    "settings":    {"settings","profile","privacy","account","theme","language","backup",
                    "sync","storage"},
    "permission":  {"permission","camera","contacts","location","microphone","allow","deny",
                    "granted","revoked"},
    "crash":       {"crash","freeze","hang","stuck","lag","slow","anr","unresponsive",
                    "force_close","not_responding","black_screen","white_screen"},
    "payment":     {"payment","purchase","subscription","billing","invoice","refund","card",
                    "wallet","topup","transfer","transaction"},
    "ui":          {"menu","overflow","kebab","tab","button","tap","click","press",
                    "longpress","swipe","scroll","open","close","back","gesture",
                    "layout","overlap","misalign","truncate","cut","hidden"},
}

# Flat intent set (for quick lookup)
INTENT_KEYWORDS: Set[str] = {kw for grp in FLOW_GROUPS.values() for kw in grp}

STOPWORDS = set(
    """
a an the and or but if then else when while for to of in on at by with without from into
is are was were be been being this that these those it its as
ve veya ama eğer ise değil için ile bir bu da de
""".split()
)

IGNORE_REGEXES = [
    r"\b(app\s*)?version\s*[:=]\s*[^\n\r]+",
    r"\bbuild\s*[:=]\s*[^\n\r]+",
    r"\bdevice\s*[:=]\s*[^\n\r]+",
    r"\blogs?\s*[:=]\s*[^\n\r]+",
    r"\brepro(duction)?\b.*",
    r"https?://\S+",
    r"\b\d+\.\d+\.\d+(\.\d+)?\b",
    r"\b\d+\b",
]

SENTENCE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ----------------------------
# Sentence-transformers loader (cached)
# ----------------------------
@st.cache_resource(show_spinner="Loading semantic model…")
def load_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SENTENCE_MODEL_NAME)
        return model
    except ImportError:
        st.error(
            "sentence-transformers not installed. Run:\n\n"
            "  pip install sentence-transformers\n\n"
            "Falling back to TF-IDF only mode."
        )
        return None


@st.cache_data(show_spinner="Encoding semantic embeddings…")
def encode_texts(_model, texts: Tuple[str, ...]) -> Optional[np.ndarray]:
    if _model is None:
        return None
    return _model.encode(list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True)


def semantic_cosine(emb: np.ndarray, i: int, j: int) -> float:
    """Cosine similarity from L2-normalised embeddings (dot product)."""
    return float(np.dot(emb[i], emb[j]))


# ----------------------------
# Helpers
# ----------------------------
def pick_col(cols: List[str], must_contain_any: List[str]) -> Optional[str]:
    for c in cols:
        cl = c.lower().strip()
        if any(k in cl for k in must_contain_any):
            return c
    return None


def normalize_text(summary, desc) -> str:
    s = f"{'' if pd.isna(summary) else str(summary)} {'' if pd.isna(desc) else str(desc)}"
    s = s.lower()
    for pat in IGNORE_REGEXES:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(norm: str) -> List[str]:
    return [t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS]


def jaccard(a_set: Set[str], b_set: Set[str]) -> float:
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def get_flows(token_set: Set[str]) -> Set[str]:
    """Return which user flows are present in this issue."""
    flows = set()
    for flow, keywords in FLOW_GROUPS.items():
        if token_set & keywords:
            flows.add(flow)
    return flows


def flow_overlap(a_set: Set[str], b_set: Set[str]) -> Tuple[int, Set[str]]:
    fa = get_flows(a_set)
    fb = get_flows(b_set)
    shared = fa & fb
    return len(shared), shared


def parse_created(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def determine_keep_delete(i: int, j: int, created_dt: Optional[pd.Series]) -> Tuple[int, int]:
    if created_dt is not None:
        ci, cj = created_dt.iloc[i], created_dt.iloc[j]
        if pd.notna(ci) and pd.notna(cj):
            return (i, j) if ci <= cj else (j, i)
    return (i, j) if i < j else (j, i)


def df_to_section_csv(df: Optional[pd.DataFrame], title: str) -> str:
    header = f"# {title}\n"
    if df is None or df.empty:
        return header + "(empty)\n\n"
    return header + df.to_csv(index=False) + "\n"


# ----------------------------
# Data prep / TF-IDF neighbors
# ----------------------------
@st.cache_data(show_spinner=False)
def load_and_preprocess(raw_csv: str):
    df = pd.read_csv(StringIO(raw_csv), sep=None, engine="python", on_bad_lines="skip")
    cols = list(df.columns)

    key_col     = pick_col(cols, ["issue key", "issue_key", "key"]) or cols[0]
    summary_col = pick_col(cols, ["summary", "title"]) or cols[0]
    desc_col    = pick_col(cols, ["description"]) or cols[0]
    created_col = pick_col(cols, ["created", "created date", "created_at", "createdat"])

    work = df.copy()
    work["_key"]       = work[key_col].astype(str).str.strip()
    work["_norm"]      = [normalize_text(a, b) for a, b in zip(work[summary_col], work[desc_col])]
    work["_tokens"]    = work["_norm"].apply(tokenize)
    work["_set"]       = work["_tokens"].apply(set)
    work["_tok_count"] = work["_tokens"].apply(len)

    # Raw text for semantic embedding (summary + description, no noise stripping)
    work["_raw_text"]  = [
        f"{'' if pd.isna(a) else str(a)} {'' if pd.isna(b) else str(b)}".strip()
        for a, b in zip(work[summary_col], work[desc_col])
    ]

    created_dt = parse_created(work[created_col]) if created_col else None

    detected = {
        "Issue Key":            key_col,
        "Summary/Title":        summary_col,
        "Description":          desc_col,
        "Created (optional)":   created_col if created_col else "(not found)",
    }
    return work, created_dt, detected


@st.cache_data(show_spinner="Building TF-IDF candidate index…")
def tfidf_topk_neighbors(texts: List[str], topk: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(
        n_neighbors=min(topk + 1, X.shape[0]),
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    return indices[:, 1:], 1.0 - distances[:, 1:]


def connected_components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        if a != b:
            adj[a].append(b)
            adj[b].append(a)
    seen = [False] * n
    comps = []
    for i in range(n):
        if seen[i] or not adj[i]:
            continue
        q = deque([i])
        seen[i] = True
        comp = [i]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        if len(comp) >= 2:
            comps.append(sorted(comp))
    return comps


# ----------------------------
# Core pipeline
# ----------------------------
def run_pipeline(work: pd.DataFrame, created_dt: Optional[pd.Series], emb: Optional[np.ndarray]):
    n = len(work)
    use_semantic = emb is not None

    # 1) TF-IDF candidate pairs
    nn_idx, nn_sim = tfidf_topk_neighbors(work["_norm"].tolist(), TOP_K_NEIGHBORS)

    cand_best: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for pos in range(nn_idx.shape[1]):
            j = int(nn_idx[i, pos])
            cos = float(nn_sim[i, pos])
            if cos < CANDIDATE_COS_THRESHOLD:
                continue
            a, b = (i, j) if i < j else (j, i)
            if a == b:
                continue
            if (a, b) not in cand_best or cos > cand_best[(a, b)]:
                cand_best[(a, b)] = cos

    candidate_pairs = sorted(cand_best.items(), key=lambda x: x[1], reverse=True)

    # 2) Score + classify each candidate pair
    deleted_already: Set[int] = set()
    out_rows: List[Dict] = []
    edges_for_cluster: List[Tuple[int, int]] = []

    for (a, b), tfidf_cos in candidate_pairs:
        keep_idx, del_idx = determine_keep_delete(a, b, created_dt)
        if del_idx in deleted_already:
            continue

        set_keep = work["_set"].iloc[keep_idx]
        set_del  = work["_set"].iloc[del_idx]
        norm_keep = work["_norm"].iloc[keep_idx]
        norm_del  = work["_norm"].iloc[del_idx]

        tok_keep = int(work["_tok_count"].iloc[keep_idx])
        tok_del  = int(work["_tok_count"].iloc[del_idx])

        shared   = len(set_keep & set_del)
        jac      = jaccard(set_keep, set_del)

        # Flow / scenario overlap
        n_flows, shared_flows = flow_overlap(set_keep, set_del)

        # Semantic embedding cosine (if available)
        sem_cos = semantic_cosine(emb, keep_idx, del_idx) if use_semantic else None

        has_evidence = (tok_keep >= SAFE_TOKEN_MIN and tok_del >= SAFE_TOKEN_MIN)
        has_flow     = (n_flows >= 1)

        # ── Decision logic ──────────────────────────────────────────────
        exact_safe = (norm_keep == norm_del) and has_evidence and has_flow

        # Token-level strict safe (original logic, requires flow overlap now)
        token_safe = (
            has_evidence
            and has_flow
            and jac >= SAFE_JACCARD_THRESHOLD
            and shared >= SAFE_MIN_SHARED_TOKENS
            and tfidf_cos >= SAFE_MIN_SEMANTIC_COSINE
        )

        # Semantic embedding safe (new — catches paraphrase / same flow different words)
        emb_safe = (
            use_semantic
            and has_flow
            and sem_cos is not None
            and sem_cos >= SEMANTIC_EMB_SAFE_THRESHOLD
        )

        # Semantic embedding review (catches same flow, moderate similarity)
        emb_review = (
            use_semantic
            and has_flow
            and sem_cos is not None
            and sem_cos >= SEMANTIC_EMB_REVIEW_THRESHOLD
        )

        if exact_safe:
            decision  = "SAFEDELETESTRICT"
            dup_type  = "EXACT"
            sim_out   = 1.000
            deleted_already.add(del_idx)
        elif token_safe or emb_safe:
            decision  = "SAFEDELETESTRICT"
            dup_type  = "SEMANTIC" if token_safe else "SEMANTIC_EMB"
            sim_out   = jac if token_safe else (sem_cos or tfidf_cos)
            deleted_already.add(del_idx)
        elif emb_review or tfidf_cos >= CANDIDATE_COS_THRESHOLD:
            decision  = "QA_REVIEW"
            dup_type  = "SEMANTIC_EMB" if (use_semantic and emb_review) else "SEMANTIC"
            sim_out   = sem_cos if (use_semantic and sem_cos is not None) else tfidf_cos
        else:
            continue  # below all thresholds after semantic check — skip

        out_rows.append({
            "Issue Key (Keep)":   work["_key"].iloc[keep_idx],
            "Issue Key (Delete)": work["_key"].iloc[del_idx],
            "Duplicate Type":     dup_type,
            "Similarity":         round(float(sim_out), 3),
            "Decision":           decision,
            "Shared Flows":       ", ".join(sorted(shared_flows)) if shared_flows else "—",
            # Diagnostics
            "TF-IDF Cosine":      round(float(tfidf_cos), 3),
            "Semantic Cosine":    round(float(sem_cos), 3) if sem_cos is not None else "N/A",
            "Jaccard":            round(float(jac), 3),
            "SharedTokens":       int(shared),
            "TokCountKeep":       int(tok_keep),
            "TokCountDelete":     int(tok_del),
        })
        edges_for_cluster.append((keep_idx, del_idx))

    out_df = pd.DataFrame(out_rows)

    if not out_df.empty:
        order = {"SAFEDELETESTRICT": 0, "QA_REVIEW": 1}
        out_df["_ord"] = out_df["Decision"].map(order).fillna(9)
        out_df = (
            out_df
            .sort_values(["_ord", "Similarity"], ascending=[True, False])
            .drop(columns=["_ord"])
        )

    safe_df   = out_df[out_df["Decision"] == "SAFEDELETESTRICT"].copy() if not out_df.empty else pd.DataFrame()
    review_df = out_df[out_df["Decision"] == "QA_REVIEW"].copy()        if not out_df.empty else pd.DataFrame()

    # Clusters
    clusters = connected_components(n, edges_for_cluster) if edges_for_cluster else []
    cluster_df = None
    if clusters:
        cluster_rows = []
        for cid, comp in enumerate(clusters, start=1):
            members = [work["_key"].iloc[i] for i in comp]
            flows   = set()
            for i in comp:
                flows |= get_flows(work["_set"].iloc[i])
            cluster_rows.append({
                "Cluster":      cid,
                "Size":         len(comp),
                "Flows":        ", ".join(sorted(flows)),
                "Members":      ", ".join(members),
            })
        cluster_df = pd.DataFrame(cluster_rows).sort_values(
            ["Size", "Cluster"], ascending=[False, True]
        )

    # Summary
    summary_df = pd.DataFrame([
        {"Metric": "Total issues",                   "Value": int(n)},
        {"Metric": "Semantic model",                 "Value": SENTENCE_MODEL_NAME if use_semantic else "TF-IDF only"},
        {"Metric": "TF-IDF candidate threshold",     "Value": CANDIDATE_COS_THRESHOLD},
        {"Metric": "Top-K neighbors",                "Value": TOP_K_NEIGHBORS},
        {"Metric": "Emb SAFE threshold",             "Value": SEMANTIC_EMB_SAFE_THRESHOLD if use_semantic else "N/A"},
        {"Metric": "Emb REVIEW threshold",           "Value": SEMANTIC_EMB_REVIEW_THRESHOLD if use_semantic else "N/A"},
        {"Metric": "SAFE token Jaccard threshold",   "Value": SAFE_JACCARD_THRESHOLD},
        {"Metric": "SAFE min shared tokens",         "Value": SAFE_MIN_SHARED_TOKENS},
        {"Metric": "SAFE TF-IDF cosine guard",       "Value": SAFE_MIN_SEMANTIC_COSINE},
        {"Metric": "Flow overlap required",          "Value": "YES"},
        {"Metric": "SAFEDELETESTRICT count",         "Value": int(len(safe_df))   if not safe_df.empty   else 0},
        {"Metric": "QA_REVIEW count",                "Value": int(len(review_df)) if not review_df.empty else 0},
        {"Metric": "Total flagged",                  "Value": int(len(out_df))    if not out_df.empty    else 0},
        {"Metric": "Cluster count",                  "Value": int(len(cluster_df)) if cluster_df is not None else 0},
    ])

    display_cols = ["Issue Key (Keep)", "Issue Key (Delete)", "Duplicate Type",
                    "Similarity", "Shared Flows", "Decision"]
    safe_display   = safe_df[display_cols].copy()   if not safe_df.empty   else pd.DataFrame(columns=display_cols)
    review_display = review_df[display_cols].copy() if not review_df.empty else pd.DataFrame(columns=display_cols)

    return summary_df, safe_display, review_display, cluster_df, out_df


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(
        page_title="Defect Duplicate Detector — Semantic",
        layout="wide",
    )
    st.title("🔍 Defect Duplicate Detector — Semantic + Flow Aware")

    st.markdown(
        """
        **Pipeline:**  
        1. TF-IDF cosine → candidate pairs  
        2. `sentence-transformers` multilingual embedding → semantic similarity  
        3. User-flow grouping (auth, messaging, calling, …) → scenario match guard  
        4. Combined scoring → **SAFEDELETESTRICT** / **QA_REVIEW**
        """
    )

    # Load model eagerly so user sees progress
    model = load_sentence_model()
    if model is None:
        st.warning("Running in TF-IDF only mode — install sentence-transformers for full semantic detection.")

    uploaded = st.file_uploader("Upload defects CSV", type=["csv"])
    if not uploaded:
        st.stop()

    raw = uploaded.getvalue().decode("utf-8", errors="replace")
    work, created_dt, detected = load_and_preprocess(raw)

    with st.expander("Detected columns", expanded=False):
        st.json(detected)

    # Encode with semantic model
    raw_texts = tuple(work["_raw_text"].tolist())
    emb = encode_texts(model, raw_texts) if model is not None else None

    with st.spinner("Running duplicate detection pipeline…"):
        summary_df, safe_view, review_view, cluster_df, full_df = run_pipeline(work, created_dt, emb)

    # ── Results ──────────────────────────────────────────────────────────
    st.subheader("📊 Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        safe_count   = len(safe_view)   if not safe_view.empty   else 0
        review_count = len(review_view) if not review_view.empty else 0
        st.metric("SAFEDELETESTRICT", safe_count)
    with col2:
        st.metric("QA_REVIEW", review_count)

    st.subheader("✅ SAFEDELETESTRICT")
    st.caption("These pairs share the same user flow AND exceed the semantic similarity threshold.")
    st.dataframe(safe_view, use_container_width=True, hide_index=True)

    st.subheader("🔎 QA_REVIEW")
    st.caption("Possibly duplicate — same flow detected but similarity below auto-delete threshold.")
    st.dataframe(review_view, use_container_width=True, hide_index=True)

    st.subheader("🕸️ Clusters (Connected Components)")
    st.caption("Issues connected by at least one duplicate edge, annotated with detected user flows.")
    if cluster_df is None:
        st.write("(empty)")
    else:
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    # ── Downloads ────────────────────────────────────────────────────────
    st.divider()
    combined_csv  = df_to_section_csv(safe_view,   "SAFEDELETESTRICT")
    combined_csv += df_to_section_csv(review_view, "QA_REVIEW")
    combined_csv += df_to_section_csv(cluster_df,  "CLUSTERS")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download ALL (SAFE + QA + CLUSTERS)",
            data=combined_csv.encode("utf-8"),
            file_name="defect_duplicates_all.csv",
            mime="text/csv",
        )
    with dl2:
        st.download_button(
            "⬇️ Download Full Diagnostics",
            data=full_df.to_csv(index=False).encode("utf-8") if full_df is not None else b"",
            file_name="defect_duplicates_diagnostics.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
