

import re
from collections import deque
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st


# ----------------------------
# Config (LATEST)
# ----------------------------
# Candidate generation (recall)
CANDIDATE_COS_THRESHOLD = 0.65
TOP_K_NEIGHBORS = 50

# SAFEDELETESTRICT (precision)
SAFE_JACCARD_THRESHOLD = 0.80
SAFE_MIN_SHARED_TOKENS = 5
SAFE_MIN_SEMANTIC_COSINE = 0.78  # semantic agreement guard (TF-IDF cosine)

# Stopwords (light)
STOPWORDS = set(
    """
a an the and or but if then else when while for to of in on at by with without from into
is are was were be been being this that these those it its as
ve veya ama eğer ise değil için ile
""".split()
)

# Ignore: log/version/repro fields per your rule
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
login logout register signup sign-in sign-in otp verification verify password pin biometrics fingerprint faceid
message chat send receive delivery delivered read unread typing sticker emoji gif media photo video file document
call voice video_call videocall ringing ring answer decline reject missed mute speaker bluetooth headset mic microphone
notification push badge sound vibration
channel discovery explore search
story status
permission camera contacts storage location
crash freeze hang stuck lag slow anr
error failed fail cannot can't unable wont won't
""".lower().replace("video_call", "videocall").split()
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
    # map some variants
    mapped = set()
    for t in token_set:
        if t == "video_call":
            mapped.add("videocall")
        else:
            mapped.add(t)
    return {t for t in mapped if t in INTENT_KEYWORDS}


def intent_overlap_ok(a_set: Set[str], b_set: Set[str]) -> bool:
    ia = intent_tokens(a_set)
    ib = intent_tokens(b_set)
    return len(ia & ib) >= 1


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
    work["_set"] = work["_norm"].apply(lambda x: set(tokenize(x)))

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
    """
    TF-IDF vectors + NearestNeighbors(cosine).
    Returns:
      nn_idx: [n, topk] neighbor indices
      nn_sim: [n, topk] cosine similarity
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform(texts)

    nn = NearestNeighbors(n_neighbors=min(topk + 1, X.shape[0]), metric="cosine", algorithm="brute")
    nn.fit(X)

    distances, indices = nn.kneighbors(X)
    indices = indices[:, 1:]       # drop self
    sims = 1.0 - distances[:, 1:]  # cosine similarity
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

    # 1) EXACT duplicates (normalized equality)
    seen_norm: Dict[str, int] = {}
    exact_pairs: List[Tuple[int, int]] = []
    exact_deleted: Set[int] = set()

    for i, nm in enumerate(work["_norm"]):
        if not nm:
            continue
        if nm in seen_norm:
            j = seen_norm[nm]
            keep_idx, del_idx = determine_keep_delete(i, j, created_dt)
            exact_pairs.append((keep_idx, del_idx))
            exact_deleted.add(del_idx)
        else:
            seen_norm[nm] = i

    # 2) Candidate generation via TF-IDF cosine
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

    # 3) Decide SAFEDELETESTRICT vs QA_REVIEW
    safe_deleted: Set[int] = set(exact_deleted)

    out_rows: List[Dict] = []
    edges_for_cluster: List[Tuple[int, int]] = []

    # EXACT -> SAFE
    for keep_idx, del_idx in exact_pairs:
        out_rows.append(
            {
                "Issue Key (Keep)": work["_key"].iloc[keep_idx],
                "Issue Key (Delete)": work["_key"].iloc[del_idx],
                "Duplicate Type": "EXACT",
                "Similarity": 1.000,
                "Decision": "SAFEDELETESTRICT",
            }
        )
        edges_for_cluster.append((keep_idx, del_idx))

    # SEMANTIC candidates
    for a, b, cos in candidate_pairs:
        keep_idx, del_idx = determine_keep_delete(a, b, created_dt)
        if del_idx in safe_deleted:
            continue

        sa = work["_set"].iloc[keep_idx]
        sb = work["_set"].iloc[del_idx]
        shared = len(sa & sb)
        jac = jaccard(sa, sb)

        # LATEST SAFEDELETESTRICT:
        #   must pass ALL:
        #     - strong overlap
        #     - intent overlap
        #     - semantic agreement (cosine high enough)
        is_safe = (
            shared >= SAFE_MIN_SHARED_TOKENS
            and jac >= SAFE_JACCARD_THRESHOLD
            and intent_overlap_ok(sa, sb)
            and cos >= SAFE_MIN_SEMANTIC_COSINE
        )

        if is_safe:
            decision = "SAFEDELETESTRICT"
            dup_type = "SEMANTIC"
            sim_out = jac  # deterministic + conservative for SAFE
            safe_deleted.add(del_idx)
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
            }
        )
        edges_for_cluster.append((keep_idx, del_idx))

    out_df = pd.DataFrame(out_rows)

    # Sort: SAFE first, then QA; within, higher similarity first
    if not out_df.empty:
        order = {"SAFEDELETESTRICT": 0, "QA_REVIEW": 1}
        out_df["_ord"] = out_df["Decision"].map(order).fillna(9)
        out_df = out_df.sort_values(["_ord", "Similarity"], ascending=[True, False]).drop(columns=["_ord"])

    safe_df = (
        out_df[out_df["Decision"] == "SAFEDELETESTRICT"].copy()
        if not out_df.empty
        else pd.DataFrame(columns=["Issue Key (Keep)", "Issue Key (Delete)", "Duplicate Type", "Similarity", "Decision"])
    )
    review_df = (
        out_df[out_df["Decision"] == "QA_REVIEW"].copy()
        if not out_df.empty
        else pd.DataFrame(columns=["Issue Key (Keep)", "Issue Key (Delete)", "Duplicate Type", "Similarity", "Decision"])
    )

    # 4) Clustering (connected components on all flagged edges)
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
            {"Metric": "SAFE: Jaccard threshold", "Count": SAFE_JACCARD_THRESHOLD},
            {"Metric": "SAFE: shared tokens", "Count": SAFE_MIN_SHARED_TOKENS},
            {"Metric": "SAFE: semantic cosine guard", "Count": SAFE_MIN_SEMANTIC_COSINE},
            {"Metric": "SAFE: intent overlap required", "Count": "YES"},
            {"Metric": "Exact duplicate count", "Count": int((safe_df["Duplicate Type"] == "EXACT").sum()) if not safe_df.empty else 0},
            {"Metric": "SAFEDELETESTRICT", "Count": int(len(safe_df))},
            {"Metric": "QA_REVIEW", "Count": int(len(review_df))},
            {"Metric": "Total flagged", "Count": int(len(out_df))},
            {"Metric": "Cluster count", "Count": int(len(cluster_df)) if cluster_df is not None else 0},
        ]
    )

    return summary_df, safe_df, review_df, cluster_df


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="BSDV_CLEAN_DEFECT_HYBRID (Latest)", layout="wide")
    st.title("BSDV_CLEAN_DEFECT_HYBRID (Latest)")

    uploaded = st.file_uploader("Upload defects CSV", type=["csv"], key="uploader_defects_csv")
    if not uploaded:
        st.stop()

    raw = uploaded.getvalue().decode("utf-8", errors="replace")
    work, created_dt, detected = load_and_preprocess(raw)

    st.write("Detected columns:")
    st.json(detected)

    with st.spinner("Running pipeline..."):
        summary_df, safe_df, review_df, cluster_df = run_pipeline(work, created_dt)

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("SAFEDELETESTRICT")
    st.dataframe(safe_df, use_container_width=True, hide_index=True)

    st.subheader("QA_REVIEW")
    st.dataframe(review_df, use_container_width=True, hide_index=True)

    st.subheader("CLUSTERS (Connected Components)")
    if cluster_df is None:
        st.write("(empty)")
    else:
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    # One CSV download with 3 sections
    combined_csv = ""
    combined_csv += df_to_section_csv(safe_df, "SAFEDELETESTRICT")
    combined_csv += df_to_section_csv(review_df, "QA_REVIEW")
    combined_csv += df_to_section_csv(cluster_df, "CLUSTERS")

    st.download_button(
        "Download ALL (SAFE + QA_REVIEW + CLUSTERS) as ONE CSV",
        data=combined_csv.encode("utf-8"),
        file_name="BSDV_CLEAN_DEFECT_HYBRID_ALL.csv",
        mime="text/csv",
        key="dl_all_one",
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
