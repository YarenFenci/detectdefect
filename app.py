
import re
from collections import defaultdict
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# BSDV SAFE config (locked)
# ----------------------------
BSDV_MIN_SHARED_TOKENS = 5
BSDV_SAFE_JACCARD = 0.80  # SAFE_DELETE_STRICT semantic threshold (unchanged)

# QA (fast) defaults
QA_COS_THRESHOLD_DEFAULT = 0.86
QA_TOPK_DEFAULT = 10

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
def preprocess_df(raw_csv: str):
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
def tfidf_topk_neighbors(texts: list[str], topk: int):
    """
    TF-IDF vectors + NearestNeighbors(cosine)
    Returns:
      indices: [n, topk]
      sims:    [n, topk] cosine similarity
      method label
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    X = vec.fit_transform(texts)

    nn = NearestNeighbors(n_neighbors=topk + 1, metric="cosine", algorithm="brute")
    nn.fit(X)

    distances, indices = nn.kneighbors(X)
    indices = indices[:, 1:]          # drop self
    sims = 1.0 - distances[:, 1:]     # cosine sim = 1 - cosine distance
    return indices, sims, "TF-IDF + NearestNeighbors(cosine) (ULTRA FAST)"


def main():
    st.set_page_config(page_title="BSDV_CLEAN_DEFECT_HYBRID (ULTRA FAST)", layout="wide")
    st.title("BSDV_CLEAN_DEFECT_HYBRID (ULTRA FAST)")
    st.caption("SAFE_DELETE_STRICT = BSDV (unchanged), QA_REVIEW = TF-IDF cosine top-k neighbors.")

    uploaded = st.file_uploader("Upload defects CSV", type=["csv"])

    with st.expander("Rules", expanded=False):
        st.markdown(
            f"""
**SAFE_DELETE_STRICT (BSDV):**
- EXACT: normalized text identical
- SEMANTIC SAFE: shared tokens ≥ {BSDV_MIN_SHARED_TOKENS} AND Jaccard ≥ {BSDV_SAFE_JACCARD}

**QA_REVIEW (ULTRA FAST):**
- TF-IDF + NearestNeighbors(cosine)
- cosine similarity ≥ threshold (default {QA_COS_THRESHOLD_DEFAULT})
- only top-K neighbors per issue (default {QA_TOPK_DEFAULT})
- excludes SAFE_DELETE_STRICT
"""
        )

    run_qa = st.checkbox("Run QA_REVIEW (fast)", value=True)
    qa_threshold = st.slider("QA_REVIEW cosine threshold", 0.70, 0.99, QA_COS_THRESHOLD_DEFAULT, 0.01)
    qa_topk = st.slider("QA_REVIEW top-k neighbors", 3, 30, QA_TOPK_DEFAULT, 1)

    if not uploaded:
        st.stop()

    raw = uploaded.getvalue().decode("utf-8", errors="replace")
    work, created_dt, detected = preprocess_df(raw)

    st.write("Detected columns:")
    st.json(detected)

    # ----------------------------
    # SAFE_DELETE_STRICT (BSDV)
    # ----------------------------
    # EXACT
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

    # token inverted index for semantic SAFE
    inv = defaultdict(list)
    for i, s in enumerate(work["_set"]):
        for t in s:
            inv[t].append(i)

    safe_semantic_pairs = []
    safe_semantic_delete_idx = set()
    seen_pair = set()

    for i, si in enumerate(work["_set"]):
        if i in exact_delete_idx or i in safe_semantic_delete_idx:
            continue

        candidates = set()
        for t in si:
            candidates.update(inv[t])

        for j in candidates:
            if j == i:
                continue
            a, b = (j, i) if j < i else (i, j)
            if (a, b) in seen_pair:
                continue
            seen_pair.add((a, b))

            sj = work["_set"].iloc[j]
            if len(si & sj) < BSDV_MIN_SHARED_TOKENS:
                continue

            score = jaccard(si, sj)
            if score >= BSDV_SAFE_JACCARD:
                keep_idx, del_idx = determine_keep_delete(i, j, created_dt)
                if del_idx in exact_delete_idx or del_idx in safe_semantic_delete_idx:
                    continue
                safe_semantic_pairs.append((keep_idx, del_idx, score))
                safe_semantic_delete_idx.add(del_idx)

    safe_delete_set = set([d for _, d in exact_pairs] + [d for _, d, _ in safe_semantic_pairs])

    # ----------------------------
    # QA_REVIEW (TF-IDF top-k)
    # ----------------------------
    qa_pairs = []
    qa_method = "disabled"

    if run_qa:
        with st.spinner("Finding QA_REVIEW candidates (TF-IDF top-k neighbors)..."):
            nn_idx, nn_sim, qa_method = tfidf_topk_neighbors(work["_norm"].tolist(), qa_topk)

        qa_candidate_rows = []
        for i in range(len(work)):
            for pos in range(nn_idx.shape[1]):
                j = int(nn_idx[i, pos])
                score = float(nn_sim[i, pos])

                if score < qa_threshold:
                    continue

                keep_idx, del_idx = determine_keep_delete(i, j, created_dt)
                if del_idx in safe_delete_set:
                    continue

                pair = (min(keep_idx, del_idx), max(keep_idx, del_idx))
                qa_candidate_rows.append((keep_idx, del_idx, score, pair))

        # keep best score per pair
        best = {}
        for keep_idx, del_idx, score, pair in qa_candidate_rows:
            if pair not in best or score > best[pair][2]:
                best[pair] = (keep_idx, del_idx, score)

        qa_pairs = list(best.values())
        qa_pairs.sort(key=lambda x: x[2], reverse=True)

    # ----------------------------
    # Combined Output
    # ----------------------------
    rows = []

    for keep_idx, del_idx in exact_pairs:
        rows.append(
            {
                "Issue Key (Keep)": work["_key"].iloc[keep_idx],
                "Issue Key (Delete)": work["_key"].iloc[del_idx],
                "Duplicate Type": "EXACT",
                "Similarity": 1.000,
                "Decision": "SAFE_DELETE_STRICT",
                "Method": "BSDV exact",
            }
        )

    for keep_idx, del_idx, score in safe_semantic_pairs:
        rows.append(
            {
                "Issue Key (Keep)": work["_key"].iloc[keep_idx],
                "Issue Key (Delete)": work["_key"].iloc[del_idx],
                "Duplicate Type": "SEMANTIC",
                "Similarity": round(float(score), 3),
                "Decision": "SAFE_DELETE_STRICT",
                "Method": "BSDV semantic (Jaccard)",
            }
        )

    for keep_idx, del_idx, score in qa_pairs:
        rows.append(
            {
                "Issue Key (Keep)": work["_key"].iloc[keep_idx],
                "Issue Key (Delete)": work["_key"].iloc[del_idx],
                "Duplicate Type": "SEMANTIC",
                "Similarity": round(float(score), 3),
                "Decision": "QA_REVIEW",
                "Method": qa_method,
            }
        )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        order = {"SAFE_DELETE_STRICT": 0, "QA_REVIEW": 1}
        out_df["_ord"] = out_df["Decision"].map(order).fillna(9)
        out_df = out_df.sort_values(["_ord", "Similarity"], ascending=[True, False]).drop(columns=["_ord"])

    summary_df = pd.DataFrame(
        [
            {"Metric": "Total issue count", "Count": len(work)},
            {"Metric": "SAFE_DELETE_STRICT", "Count": int((out_df["Decision"] == "SAFE_DELETE_STRICT").sum()) if not out_df.empty else 0},
            {"Metric": "QA_REVIEW", "Count": int((out_df["Decision"] == "QA_REVIEW").sum()) if not out_df.empty else 0},
            {"Metric": "QA_REVIEW method", "Count": qa_method},
            {"Metric": "QA_REVIEW cosine threshold", "Count": qa_threshold if run_qa else "-"},
            {"Metric": "QA_REVIEW top-k", "Count": qa_topk if run_qa else "-"},
        ]
    )

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Combined Output")
    st.dataframe(out_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Combined CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="BSDV_CLEAN_DEFECT_HYBRID_ULTRAFAST_OUTPUT.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download Summary CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="BSDV_CLEAN_DEFECT_HYBRID_ULTRAFAST_SUMMARY.csv",
        mime="text/csv",
    )

    st.caption("QA_REVIEW items must be validated for same root cause before deletion.")


if __name__ == "__main__":
    main()
