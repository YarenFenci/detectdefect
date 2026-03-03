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
CANDIDATE_COS_THRESHOLD   = 0.60    # TF-IDF pre-filter
TOP_K_NEIGHBORS           = 50
SAFE_TOKEN_MIN            = 15

# Token-level guard
SAFE_JACCARD_THRESHOLD    = 0.88
SAFE_MIN_SHARED_TOKENS    = 10
SAFE_MIN_TFIDF_COSINE     = 0.82

# Embedding threshold — single high-confidence gate
SEMANTIC_EMB_THRESHOLD    = 0.86
MIN_SIMILARITY_OUTPUT     = 0.75    # absolute floor

SENTENCE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# User-flow scenario groups
FLOW_GROUPS: Dict[str, Set[str]] = {
    "auth":         {"login","logout","signin","signup","register","otp","verification","verify",
                     "password","pin","biometrics","fingerprint","faceid","2fa","authentication"},
    "messaging":    {"message","chat","send","receive","delivery","delivered","read","unread",
                     "typing","sticker","emoji","gif","media","photo","video","file","document",
                     "forward","reply","delete","unsend","attachment"},
    "calling":      {"call","voice","videocall","video","ringing","ring","answer","decline",
                     "reject","missed","mute","speaker","bluetooth","headset","mic","microphone",
                     "call_quality","echo","noise"},
    "notification": {"notification","push","badge","sound","vibration","alert","banner"},
    "channel":      {"channel","discovery","explore","search","find","broadcast"},
    "story":        {"story","status","highlight","viewer","reaction"},
    "settings":     {"settings","profile","privacy","account","theme","language","backup",
                     "sync","storage"},
    "permission":   {"permission","camera","contacts","location","microphone","allow","deny",
                     "granted","revoked"},
    "crash":        {"crash","freeze","hang","stuck","lag","slow","anr","unresponsive",
                     "force_close","not_responding","black_screen","white_screen"},
    "payment":      {"payment","purchase","subscription","billing","invoice","refund","card",
                     "wallet","topup","transfer","transaction"},
    "ui":           {"menu","overflow","kebab","tab","button","tap","click","press",
                     "longpress","swipe","scroll","open","close","back","gesture",
                     "layout","overlap","misalign","truncate","cut","hidden"},
}

STOPWORDS = set(
    "a an the and or but if then else when while for to of in on at by with without "
    "from into is are was were be been being this that these those it its as "
    "ve veya ama eger ise degil icin ile bir bu da de".split()
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


# ----------------------------
# Sentence-transformers
# ----------------------------
@st.cache_resource(show_spinner="Loading semantic model...")
def load_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(SENTENCE_MODEL_NAME)
    except ImportError:
        st.error("sentence-transformers not installed.\n\npip install sentence-transformers")
        return None


@st.cache_data(show_spinner="Encoding embeddings...")
def encode_texts(_model, texts: Tuple[str, ...]) -> Optional[np.ndarray]:
    if _model is None:
        return None
    return _model.encode(
        list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )


def cosine_emb(emb: np.ndarray, i: int, j: int) -> float:
    return float(np.dot(emb[i], emb[j]))


# ----------------------------
# Helpers
# ----------------------------
def pick_col(cols: List[str], keywords: List[str]) -> Optional[str]:
    for c in cols:
        if any(k in c.lower() for k in keywords):
            return c
    return None


def normalize_text(summary, desc) -> str:
    s = f"{'' if pd.isna(summary) else str(summary)} {'' if pd.isna(desc) else str(desc)}"
    s = s.lower()
    for pat in IGNORE_REGEXES:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(norm: str) -> List[str]:
    return [t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS]


def jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0


def get_flows(token_set: Set[str]) -> Set[str]:
    return {flow for flow, kws in FLOW_GROUPS.items() if token_set & kws}


def flow_overlap_count(a: Set[str], b: Set[str]) -> int:
    return len(get_flows(a) & get_flows(b))


def parse_created(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def keep_delete(i: int, j: int, created_dt: Optional[pd.Series]) -> Tuple[int, int]:
    if created_dt is not None:
        ci, cj = created_dt.iloc[i], created_dt.iloc[j]
        if pd.notna(ci) and pd.notna(cj):
            return (i, j) if ci <= cj else (j, i)
    return (i, j) if i < j else (j, i)


def df_to_csv_section(df: Optional[pd.DataFrame], title: str) -> str:
    header = f"# {title}\n"
    if df is None or df.empty:
        return header + "(empty)\n\n"
    return header + df.to_csv(index=False) + "\n"


# ----------------------------
# Data prep
# ----------------------------
@st.cache_data(show_spinner=False)
def load_and_preprocess(raw_csv: str):
    df = pd.read_csv(StringIO(raw_csv), sep=None, engine="python", on_bad_lines="skip")
    cols = list(df.columns)

    key_col     = pick_col(cols, ["issue key", "issue_key", "key"]) or cols[0]
    summary_col = pick_col(cols, ["summary", "title"])              or cols[0]
    desc_col    = pick_col(cols, ["description"])                   or cols[0]
    created_col = pick_col(cols, ["created", "created date", "created_at", "createdat"])

    w = df.copy()
    w["_key"]       = w[key_col].astype(str).str.strip()
    w["_norm"]      = [normalize_text(a, b) for a, b in zip(w[summary_col], w[desc_col])]
    w["_tokens"]    = w["_norm"].apply(tokenize)
    w["_set"]       = w["_tokens"].apply(set)
    w["_tok_count"] = w["_tokens"].apply(len)
    w["_raw"]       = [
        f"{'' if pd.isna(a) else str(a)} {'' if pd.isna(b) else str(b)}".strip()
        for a, b in zip(w[summary_col], w[desc_col])
    ]

    created_dt = parse_created(w[created_col]) if created_col else None
    detected = {
        "Issue Key":          key_col,
        "Summary / Title":    summary_col,
        "Description":        desc_col,
        "Created (optional)": created_col or "(not found)",
    }
    return w, created_dt, detected


@st.cache_data(show_spinner="Building TF-IDF index...")
def tfidf_neighbors(texts: List[str], topk: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    X = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform(texts)
    nn = NearestNeighbors(
        n_neighbors=min(topk + 1, X.shape[0]), metric="cosine", algorithm="brute"
    ).fit(X)
    dist, idx = nn.kneighbors(X)
    return idx[:, 1:], 1.0 - dist[:, 1:]


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
    use_emb = emb is not None

    nn_idx, nn_sim = tfidf_neighbors(work["_norm"].tolist(), TOP_K_NEIGHBORS)

    # Build candidate pairs
    cand: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for pos in range(nn_idx.shape[1]):
            j   = int(nn_idx[i, pos])
            c   = float(nn_sim[i, pos])
            if c < CANDIDATE_COS_THRESHOLD:
                continue
            a, b = (i, j) if i < j else (j, i)
            if a != b and c > cand.get((a, b), 0):
                cand[(a, b)] = c

    rows:    List[Dict]          = []
    deleted: Set[int]            = set()
    edges:   List[Tuple[int,int]] = []

    for (a, b), tfidf_c in sorted(cand.items(), key=lambda x: x[1], reverse=True):
        ki, di = keep_delete(a, b, created_dt)
        if di in deleted:
            continue

        s_k  = work["_set"].iloc[ki]
        s_d  = work["_set"].iloc[di]
        n_k  = work["_norm"].iloc[ki]
        n_d  = work["_norm"].iloc[di]
        tc_k = int(work["_tok_count"].iloc[ki])
        tc_d = int(work["_tok_count"].iloc[di])

        # Must share at least one user flow
        if flow_overlap_count(s_k, s_d) < 1:
            continue

        shared = len(s_k & s_d)
        jac    = jaccard(s_k, s_d)
        sc     = cosine_emb(emb, ki, di) if use_emb else None
        has_ev = tc_k >= SAFE_TOKEN_MIN and tc_d >= SAFE_TOKEN_MIN

        # Confidence gates
        exact          = (n_k == n_d) and has_ev
        token_confident = (
            has_ev
            and jac    >= SAFE_JACCARD_THRESHOLD
            and shared >= SAFE_MIN_SHARED_TOKENS
            and tfidf_c >= SAFE_MIN_TFIDF_COSINE
        )
        emb_confident = use_emb and sc is not None and sc >= SEMANTIC_EMB_THRESHOLD

        if not (exact or token_confident or emb_confident):
            continue

        similarity = 1.0 if exact else (sc if sc is not None else jac)

        if round(float(similarity), 3) < MIN_SIMILARITY_OUTPUT:
            continue

        rows.append({
            "Issue (Keep)":      work["_key"].iloc[ki],
            "Issue (Duplicate)": work["_key"].iloc[di],
            "Similarity":        round(float(similarity), 3),
        })
        deleted.add(di)
        edges.append((ki, di))

    dup_df = pd.DataFrame(rows)
    if not dup_df.empty:
        dup_df = dup_df.sort_values("Similarity", ascending=False).reset_index(drop=True)

    # Clusters
    clusters   = connected_components(n, edges) if edges else []
    cluster_df = None
    if clusters:
        cluster_rows = []
        for cid, comp in enumerate(clusters, start=1):
            members    = [work["_key"].iloc[i] for i in comp]
            flows_all: Set[str] = set()
            for i in comp:
                flows_all |= get_flows(work["_set"].iloc[i])
            cluster_rows.append({
                "Cluster": cid,
                "Size":    len(comp),
                "Flow":    ", ".join(sorted(flows_all)),
                "Members": ", ".join(members),
            })
        cluster_df = (
            pd.DataFrame(cluster_rows)
            .sort_values(["Size", "Cluster"], ascending=[False, True])
            .reset_index(drop=True)
        )

    summary_df = pd.DataFrame([
        {"Metric": "Total issues",          "Value": int(n)},
        {"Metric": "Semantic model",        "Value": SENTENCE_MODEL_NAME if use_emb else "TF-IDF only"},
        {"Metric": "Embedding threshold",   "Value": SEMANTIC_EMB_THRESHOLD},
        {"Metric": "Min similarity output", "Value": MIN_SIMILARITY_OUTPUT},
        {"Metric": "Flow guard",            "Value": "YES — min 1 shared flow required"},
        {"Metric": "Duplicates found",      "Value": int(len(dup_df)) if not dup_df.empty else 0},
        {"Metric": "Clusters",              "Value": int(len(cluster_df)) if cluster_df is not None else 0},
    ])

    return summary_df, dup_df, cluster_df


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="Defect Duplicate Detector", layout="wide")
    st.title("Defect Duplicate Detector")

    model = load_sentence_model()
    if model is None:
        st.warning("Running in TF-IDF only mode — install sentence-transformers for full semantic detection.")

    uploaded = st.file_uploader("Upload defects CSV", type=["csv"])
    if not uploaded:
        st.stop()

    raw  = uploaded.getvalue().decode("utf-8", errors="replace")
    work, created_dt, detected = load_and_preprocess(raw)

    with st.expander("Detected columns", expanded=False):
        st.json(detected)

    emb = encode_texts(model, tuple(work["_raw"].tolist())) if model else None

    with st.spinner("Analysing..."):
        summary_df, dup_df, cluster_df = run_pipeline(work, created_dt, emb)

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Duplicate Issues")
    if dup_df.empty:
        st.info("No duplicates found above the confidence threshold.")
    else:
        st.dataframe(dup_df, use_container_width=True, hide_index=True)

    st.subheader("Clusters")
    if cluster_df is None:
        st.info("No clusters.")
    else:
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    st.divider()
    csv_out  = df_to_csv_section(dup_df,     "DUPLICATES")
    csv_out += df_to_csv_section(cluster_df, "CLUSTERS")

    st.download_button(
        "Download Results (Duplicates + Clusters)",
        data=csv_out.encode("utf-8"),
        file_name="defect_duplicates.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
