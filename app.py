

import re
from collections import deque
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st


# ----------------------------
# Config (LATEST / strict SAFE)
# ----------------------------
CANDIDATE_COS_THRESHOLD = 0.65
TOP_K_NEIGHBORS = 50

SAFE_TOKEN_MIN = 25  # evidence guard

# Semantic SAFE strict thresholds
SAFE_JACCARD_THRESHOLD = 0.85
SAFE_MIN_SHARED_TOKENS = 8
SAFE_MIN_SEMANTIC_COSINE = 0.78  # semantic agreement guard

STOPWORDS = set(
    """
a an the and or but if then else when while for to of in on at by with without from into
is are was were be been being this that these those it its as
ve veya ama eğer ise değil için ile
""".split()
)

# Ignore: log/version/repro differences (but keep feature context)
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

# Intent keywords (BiP-oriented)
INTENT_KEYWORDS = set(
    """
login logout register signup signin sign-in otp verification verify password pin biometrics fingerprint faceid
message chat send receive delivery delivered read unread typing sticker emoji gif media photo video file document
call voice videocall video-call ringing ring answer decline reject missed mute speaker bluetooth headset mic microphone
notification push badge sound vibration
channel discovery explore search
story status
settings profile privacy account
permission camera contacts storage location
crash freeze hang stuck lag slow anr
error failed fail cannot can't unable wont won't
menu overflow kebab three-dot more-button
tap click press longpress long-press swipe scroll open close back
""".lower().split()
)


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
    toks = []
    for t in norm.split():
        if len(t) < 3:
            continue
        if t in STOPWORDS:
            continue
        toks.append(t)
    return toks


def jaccard(a_set: Set[str], b_set: Set[str]) -> float:
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def intent_tokens(token_set: Set[str]) -> Set[str]:
    mapped = set()
    for t in token_set:
        if t in ("video_call", "video-call"):
            mapped.add("videocall")
        elif t in ("three", "dot", "dots"):
            # don't overfit; keep as-is (will be filtered by INTENT_KEYWORDS)
            mapped.add(t)
        else:
            mapped.add(t)
    return {t for t in mapped if t in INTENT_KEYWORDS}


def intent_overlap_count(a_set: Set[str], b_set: Set[str]) -> int:
    ia = intent_tokens(a_set)
    ib = intent_tokens(b_set)
    return len(ia & ib)


def parse_created(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def determine_keep_delete(i: int, j: int, created_dt: Optional[pd.Series]) -> Tuple[int, int]:
    # Deterministic KEEP:
    # - older created date KEEP, if both parseable
    # - else lower row index KEEP
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

    key_col = pick_col(cols, ["issue key", "issue_key", "key"]) or cols[0]
    summary_col = pick_col(cols, ["summary", "title"]) or cols[0]
    desc_col = pick_col(cols, ["description"]) or cols[0]
    created_col = pick_col(cols, ["created", "created date", "created_at", "createdat"])

    work = df.copy()
    work["_key"] = work[key_col].astype(str).str.strip()
    work["_norm"] = [normalize_text(a, b) for a, b in zip(work[summary_col], work[desc_col])]
    work["_tokens"] = work["_norm"].apply(tokenize)
    work["_set"] = work["_tokens"].apply(set)
    work["_tok_count"] = work["_tokens"].apply(len)

    created_dt = None
    if created_col:
        created_dt = parse_created(work[created_col])

    detected = {
        "Issue Key": key_col,
        "Summary/Title": summary_col,
        "Description": desc_col,
        "Created (optional)": created_col if created_col else "(not found)",
    }
    return work, created_dt, detected


@st.cache_data(show_spinner=False)
def tfidf_topk_neighbors(texts: List[str], topk: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform(texts)

    nn = NearestNeighbors(n_neighbors=min(topk + 1, X.shape[0]), metric="cosine", algorithm="brute")
    nn.fit(X)

    distances, indices = nn.kneighbors(X)
    indices = indices[:, 1:]
    sims = 1.0 - distances[:, 1:]
    return indices, sims


def connected_components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a == b:
            continue
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
def run_pipeline(work: pd.DataFrame, created_dt: Optional[pd.Series]):
    n = len(work)

    # 1) Candidate generation via TF-IDF cosine
    nn_idx, nn_sim = tfidf_topk_neighbors(work["_norm"].tolist(), TOP_K_NEIGHBORS)

    # Deduplicate candidate pairs, keep best cosine
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

    candidate_pairs = [(a, b, cand_best[(a, b)]) for (a, b) in cand_best.keys()]
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)

    # 2) Decide SAFEDELETESTRICT vs QA_REVIEW
    deleted_already: Set[int] = set()
    out_rows: List[Dict] = []
    edges_for_cluster: List[Tuple[int, int]] = []

    for a, b, cos in candidate_pairs:
        keep_idx, del_idx = determine_keep_delete(a, b, created_dt)
        if del_idx in deleted_already:
            continue

        norm_keep = work["_norm"].iloc[keep_idx]
        norm_del = work["_norm"].iloc[del_idx]

        set_keep = work["_set"].iloc[keep_idx]
        set_del = work["_set"].iloc[del_idx]

        tok_keep = int(work["_tok_count"].iloc[keep_idx])
        tok_del = int(work["_tok_count"].iloc[del_idx])

        shared = len(set_keep & set_del)
        jac = jaccard(set_keep, set_del)
        intent_shared = intent_overlap_count(set_keep, set_del)

        has_evidence = (tok_keep >= SAFE_TOKEN_MIN and tok_del >= SAFE_TOKEN_MIN)
        has_intent = (intent_shared >= 1)

        # EXACT_SAFE: identical text + evidence + intent
        exact_safe = (norm_keep == norm_del) and has_evidence and has_intent

        # SEMANTIC_SAFE: very strict + evidence + intent + semantic agreement
        semantic_safe = (
            has_evidence
            and has_intent
            and jac >= SAFE_JACCARD_THRESHOLD
            and shared >= SAFE_MIN_SHARED_TOKENS
            and cos >= SAFE_MIN_SEMANTIC_COSINE
        )

        if exact_safe or semantic_safe:
            decision = "SAFEDELETESTRICT"
            dup_type = "EXACT" if exact_safe else "SEMANTIC"
            sim_out = 1.000 if exact_safe else jac
            deleted_already.add(del_idx)
        else:
            decision = "QA_REVIEW"
            dup_type = "SEMANTIC"
            sim_out = cos

        out_rows.append(
            {
                "Issue Key (Keep)": work["_key"].iloc[keep_idx],
                "Issue Key (Delete)": work["_key"].iloc[del_idx],
                "Duplicate Type": dup_type,
                "Similarity": round(float(sim_out), 3),
                "Decision": decision,
                # Diagnostics (optional but helpful for QA review)
                "Cosine": round(float(cos), 3),
                "Jaccard": round(float(jac), 3),
                "SharedTokens": int(shared),
                "IntentShared": int(intent_shared),
                "TokCountKeep": int(tok_keep),
                "TokCountDelete": int(tok_del),
            }
        )
        edges_for_cluster.append((keep_idx, del_idx))

    out_df = pd.DataFrame(out_rows)

    # Sort
    if not out_df.empty:
        order = {"SAFEDELETESTRICT": 0, "QA_REVIEW": 1}
        out_df["_ord"] = out_df["Decision"].map(order).fillna(9)
        out_df = out_df.sort_values(["_ord", "Similarity"], ascending=[True, False]).drop(columns=["_ord"])

    safe_df = out_df[out_df["Decision"] == "SAFEDELETESTRICT"].copy() if not out_df.empty else pd.DataFrame()
    review_df = out_df[out_df["Decision"] == "QA_REVIEW"].copy() if not out_df.empty else pd.DataFrame()

    # Clusters on all flagged edges
    clusters = connected_components(n, edges_for_cluster) if edges_for_cluster else []
    cluster_df = None
    if clusters:
        cluster_rows = []
        for cid, comp in enumerate(clusters, start=1):
            members = [work["_key"].iloc[i] for i in comp]
            cluster_rows.append({"Cluster": cid, "Size": len(comp), "Members": ", ".join(members)})
        cluster_df = pd.DataFrame(cluster_rows).sort_values(["Size", "Cluster"], ascending=[False, True])

    # Summary
    summary_df = pd.DataFrame(
        [
            {"Metric": "Total issue count", "Count": int(n)},
            {"Metric": "Candidate cosine threshold", "Count": CANDIDATE_COS_THRESHOLD},
            {"Metric": "Top-K neighbors", "Count": TOP_K_NEIGHBORS},
            {"Metric": "SAFE evidence token min", "Count": SAFE_TOKEN_MIN},
            {"Metric": "SAFE semantic cosine guard", "Count": SAFE_MIN_SEMANTIC_COSINE},
            {"Metric": "SAFE semantic jaccard threshold", "Count": SAFE_JACCARD_THRESHOLD},
            {"Metric": "SAFE semantic shared token min", "Count": SAFE_MIN_SHARED_TOKENS},
            {"Metric": "SAFE intent overlap required", "Count": "YES"},
            {"Metric": "SAFEDELETESTRICT", "Count": int(len(safe_df)) if not safe_df.empty else 0},
            {"Metric": "QA_REVIEW", "Count": int(len(review_df)) if not review_df.empty else 0},
            {"Metric": "Total flagged", "Count": int(len(out_df)) if not out_df.empty else 0},
            {"Metric": "Cluster count", "Count": int(len(cluster_df)) if cluster_df is not None else 0},
        ]
    )

    # Display-only columns for SAFE/QA (as you requested earlier)
    display_cols = ["Issue Key (Keep)", "Issue Key (Delete)", "Duplicate Type", "Similarity", "Decision"]
    safe_display = safe_df[display_cols].copy() if not safe_df.empty else pd.DataFrame(columns=display_cols)
    review_display = review_df[display_cols].copy() if not review_df.empty else pd.DataFrame(columns=display_cols)

    return summary_df, safe_display, review_display, cluster_df, out_df


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="BSDV_CLEAN_DEFECT_HYBRID (Strict SAFE)", layout="wide")
    st.title("BSDV_CLEAN_DEFECT_HYBRID (Strict SAFE)")

    uploaded = st.file_uploader("Upload defects CSV", type=["csv"], key="uploader_defects_csv")
    if not uploaded:
        st.stop()

    raw = uploaded.getvalue().decode("utf-8", errors="replace")
    work, created_dt, detected = load_and_preprocess(raw)

    st.write("Detected columns:")
    st.json(detected)

    with st.spinner("Running..."):
        summary_df, safe_view, review_view, cluster_df, full_df = run_pipeline(work, created_dt)

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("SAFEDELETESTRICT")
    st.dataframe(safe_view, use_container_width=True, hide_index=True)

    st.subheader("QA_REVIEW")
    st.dataframe(review_view, use_container_width=True, hide_index=True)

    st.subheader("CLUSTERS (Connected Components)")
    if cluster_df is None:
        st.write("(empty)")
    else:
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    # ONE CSV download with 3 sections (SAFE + QA + CLUSTERS)
    combined_csv = ""
    combined_csv += df_to_section_csv(safe_view, "SAFEDELETESTRICT")
    combined_csv += df_to_section_csv(review_view, "QA_REVIEW")
    combined_csv += df_to_section_csv(cluster_df, "CLUSTERS")

    st.download_button(
        "Download ALL (SAFE + QA_REVIEW + CLUSTERS) as ONE CSV",
        data=combined_csv.encode("utf-8"),
        file_name="BSDV_CLEAN_DEFECT_ALL.csv",
        mime="text/csv",
        key="dl_all_one",
    )

    # Optional: download full diagnostics for deep QA
    st.download_button(
        "Download FULL DIAGNOSTICS CSV (optional)",
        data=full_df.to_csv(index=False).encode("utf-8") if full_df is not None else b"",
        file_name="BSDV_CLEAN_DEFECT_FULL_DIAGNOSTICS.csv",
        mime="text/csv",
        key="dl_full_diag",
    )


if __name__ == "__main__":
    main()
