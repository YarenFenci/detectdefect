

import re
from collections import defaultdict, deque
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Pipeline config (locked)
# ----------------------------
CANDIDATE_COS_THRESHOLD = 0.69
TOP_K_NEIGHBORS = 20  # coverage knob (higher = more candidates)

# Updated SAFEDELETESTRICT rule
SAFE_JACCARD_THRESHOLD = 0.75
SAFE_MIN_SHARED_TOKENS = 5
SAFE_COSINE_OVERRIDE = 0.92  # OR condition

STOPWORDS = set(
    """
a an the and or but if then else when while for to of in on at by with without from into
is are was were be been being this that these those it its as
ve veya ama eğer ise değil için ile
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


# ----------------------------
# Helpers
# ----------------------------
def pick_col(cols, must_contain_any):
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


def tokenize(norm: str):
    toks = []
    for t in norm.split():
        if len(t) < 3:
            continue
        if t in STOPWORDS:
            continue
        toks.append(t)
    return toks


def jaccard(a_set: set, b_set: set) -> float:
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def parse_created(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def determine_keep_delete(i, j, created_dt: pd.Series | None):
    # older KEEP if possible, else earlier row KEEP
    if created_dt is not None:
        ci, cj = created_dt.iloc[i], created_dt.iloc[j]
        if pd.notna(ci) and pd.notna(cj):
            return (i, j) if ci <= cj else (j, i)
    return (i, j) if i < j else (j, i)


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
def tfidf_topk_candidates(texts: list[str], topk: int):
    """
    Candidate generation with TF-IDF + NearestNeighbors(cosine).
    Returns:
      nn_idx: [n, topk] neighbor indices
      nn_sim: [n, topk] cosine similarity
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform(texts)

    nn = NearestNeighbors(n_neighbors=topk + 1, metric="cosine", algorithm="brute")
    nn.fit(X)

    distances, indices = nn.kneighbors(X)
    indices = indices[:, 1:]       # drop self
    sims = 1.0 - distances[:, 1:]  # cosine similarity
    return indices, sims


def connected_components_from_edges(n: int, edges: list[tuple[int, int]]):
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
        comps.append(sorted(comp))
    return comps


def main():
    st.set_page_config(page_title="Duplicate Detection Pipeline", layout="wide")
    st.title("Duplicate Detection Pipeline")

    uploaded = st.file_uploader("Upload defects CSV", type=["csv"])
    if not uploaded:
        st.stop()

    raw = uploaded.getvalue().decode("utf-8", errors="replace")
    work, created_dt, detected = load_and_preprocess(raw)

    st.write("Detected columns:")
    st.json(detected)

    enable_clustering = st.checkbox("Enable clustering (connected components)", value=False)

    # ----------------------------
    # EXACT duplicates (normalized text)
    # ----------------------------
    seen_norm = {}
    exact_pairs = []
    exact_delete_idx = set()

    for i, nm in enumerate(work["_norm"]):
        if not nm:
            continue
        if nm in seen_norm:
            j = seen_norm[nm]
            keep_idx, del_idx = determine_keep_delete(i, j, created_dt)
            exact_pairs.append((keep_idx, del_idx))
            exact_delete_idx.add(del_idx)
        else:
            seen_norm[nm] = i

    # ----------------------------
    # Candidate generation (TF-IDF cosine >= 0.69)
    # ----------------------------
    with st.spinner("Generating candidates with TF-IDF cosine (top-k neighbors)..."):
        nn_idx, nn_sim = tfidf_topk_candidates(work["_norm"].tolist(), TOP_K_NEIGHBORS)

    cand_best = {}  # (min_i,max_i) -> best cosine
    for i in range(len(work)):
        for pos in range(nn_idx.shape[1]):
            j = int(nn_idx[i, pos])
            score = float(nn_sim[i, pos])

            if score < CANDIDATE_COS_THRESHOLD:
                continue

            a, b = (i, j) if i < j else (j, i)
            if a == b:
                continue
            if (a, b) not in cand_best or score > cand_best[(a, b)]:
                cand_best[(a, b)] = score

    candidate_pairs = [(a, b, cand_best[(a, b)]) for (a, b) in cand_best.keys()]
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)

    # ----------------------------
    # Decision: SAFEDELETESTRICT vs QA_REVIEW
    # ----------------------------
    rows = []

    # Deterministic: prevent multiple deletes of same issue
    safe_deleted = set(exact_delete_idx)

    # EXACT -> SAFEDELETESTRICT
    for keep_idx, del_idx in exact_pairs:
        rows.append(
            {
                "Issue Key (Keep)": work["_key"].iloc[keep_idx],
                "Issue Key (Delete)": work["_key"].iloc[del_idx],
                "Duplicate Type": "EXACT",
                "Similarity": 1.000,
                "Decision": "SAFEDELETESTRICT",
            }
        )

    for a, b, cos_sim in candidate_pairs:
        keep_idx, del_idx = determine_keep_delete(a, b, created_dt)
        if del_idx in safe_deleted:
            continue

        sa = work["_set"].iloc[keep_idx]
        sb = work["_set"].iloc[del_idx]
        shared = len(sa & sb)
        jac = jaccard(sa, sb)

        # Updated rule:
        # SAFE if (jac>=0.75 AND shared>=5) OR (cos>=0.92)
        is_safe = ((shared >= SAFE_MIN_SHARED_TOKENS and jac >= SAFE_JACCARD_THRESHOLD) or (cos_sim >= SAFE_COSINE_OVERRIDE))

        if is_safe:
            decision = "SAFEDELETESTRICT"
            dup_type = "SEMANTIC"
            # Similarity: report cosine if it triggered safety override, else report jaccard
            sim_out = cos_sim if cos_sim >= SAFE_COSINE_OVERRIDE else jac
            safe_deleted.add(del_idx)
        else:
            decision = "QA_REVIEW"
            dup_type = "SEMANTIC"
            sim_out = cos_sim

        rows.append(
            {
                "Issue Key (Keep)": work["_key"].iloc[keep_idx],
                "Issue Key (Delete)": work["_key"].iloc[del_idx],
                "Duplicate Type": dup_type,
                "Similarity": round(float(sim_out), 3),
                "Decision": decision,
            }
        )

    out_df = pd.DataFrame(rows)

    if not out_df.empty:
        order = {"SAFEDELETESTRICT": 0, "QA_REVIEW": 1}
        out_df["_ord"] = out_df["Decision"].map(order).fillna(9)
        out_df = out_df.sort_values(["_ord", "Similarity"], ascending=[True, False]).drop(columns=["_ord"])

    # ----------------------------
    # Optional clustering
    # ----------------------------
    cluster_df = None
    if enable_clustering and not out_df.empty:
        key_to_idx = {k: i for i, k in enumerate(work["_key"].tolist())}
        edges = []
        for _, r in out_df.iterrows():
            ki = key_to_idx.get(r["Issue Key (Keep)"])
            kd = key_to_idx.get(r["Issue Key (Delete)"])
            if ki is None or kd is None:
                continue
            x, y = (ki, kd) if ki < kd else (kd, ki)
            edges.append((x, y))

        comps = connected_components_from_edges(len(work), edges)

        cluster_rows = []
        for cid, comp in enumerate(comps, start=1):
            keep_idx = comp[0]
            if created_dt is not None:
                comp_created = [(i, created_dt.iloc[i]) for i in comp if pd.notna(created_dt.iloc[i])]
                if comp_created:
                    keep_idx = sorted(comp_created, key=lambda x: x[1])[0][0]

            keep_key = work["_key"].iloc[keep_idx]
            members = [work["_key"].iloc[i] for i in comp]
            cluster_rows.append(
                {"Cluster": cid, "Size": len(comp), "Keep": keep_key, "Members": ", ".join(members)}
            )

        cluster_df = pd.DataFrame(cluster_rows).sort_values(["Size", "Cluster"], ascending=[False, True])

    # ----------------------------
    # Summary
    # ----------------------------
    total = len(work)
    exact_cnt = int((out_df["Duplicate Type"] == "EXACT").sum()) if not out_df.empty else 0
    safe_cnt = int((out_df["Decision"] == "SAFEDELETESTRICT").sum()) if not out_df.empty else 0
    review_cnt = int((out_df["Decision"] == "QA_REVIEW").sum()) if not out_df.empty else 0

    summary_df = pd.DataFrame(
        [
            {"Metric": "Total issue count", "Count": total},
            {"Metric": "Candidate cosine threshold", "Count": CANDIDATE_COS_THRESHOLD},
            {"Metric": "SAFE: (Jaccard >=)", "Count": SAFE_JACCARD_THRESHOLD},
            {"Metric": "SAFE: (shared tokens >=)", "Count": SAFE_MIN_SHARED_TOKENS},
            {"Metric": "SAFE: (OR cosine >=)", "Count": SAFE_COSINE_OVERRIDE},
            {"Metric": "Top-K neighbors per issue", "Count": TOP_K_NEIGHBORS},
            {"Metric": "EXACT duplicates", "Count": exact_cnt},
            {"Metric": "SAFEDELETESTRICT total", "Count": safe_cnt},
            {"Metric": "QA_REVIEW total", "Count": review_cnt},
            {"Metric": "Total flagged (safe+review)", "Count": len(out_df)},
        ]
    )

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Output")
    st.dataframe(out_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Output CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="DUPLICATE_PIPELINE_OUTPUT.csv",
        mime="text/csv",
    )

    if cluster_df is not None:
        st.subheader("Clusters (Connected Components)")
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Clusters CSV",
            data=cluster_df.to_csv(index=False).encode("utf-8"),
            file_name="DUPLICATE_PIPELINE_CLUSTERS.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
