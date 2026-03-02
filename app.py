
import re
from collections import defaultdict
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BSDV_CLEAN_DEFECT", layout="wide")

st.title("BSDV_CLEAN_DEFECT (V2)")
st.caption("Single-table output with Decision = SAFE_DELETE_STRICT / QA_REVIEW")

# ----------------------------
# Config (locked)
# ----------------------------
THRESH_MIN = 0.69
THRESH_SAFE = 0.80
MIN_SHARED_TOKENS = 5

STOPWORDS = set(
    """
a an the and or but if then else when while for to of in on at by with without from into
is are was were be been being this that these those it its as
ve veya ama eğer ise değil için ile
""".split()
)

# NOTE: These are intentionally conservative / generic.
IGNORE_REGEXES = [
    r"\b(app\s*)?version\s*[:=]\s*[^\n\r]+",
    r"\bbuild\s*[:=]\s*[^\n\r]+",
    r"\bdevice\s*[:=]\s*[^\n\r]+",
    r"\blogs?\s*[:=]\s*[^\n\r]+",
    r"\brepro(duction)?\b.*",  # user asked to ignore reproduction info if present in desc
    r"https?://\S+",
    r"\b\d+\.\d+\.\d+(\.\d+)?\b",
    r"\b\d+\b",
]

# ----------------------------
# Helpers
# ----------------------------
def pick_col(cols, must_have_any):
    """Pick first column whose lowercase name contains any token in must_have_any."""
    for c in cols:
        cl = c.lower().strip()
        if any(k in cl for k in must_have_any):
            return c
    return None


def normalize_text(summary: str, desc: str) -> str:
    s = f"{'' if pd.isna(summary) else str(summary)} {'' if pd.isna(desc) else str(desc)}"
    s = s.lower()
    for pat in IGNORE_REGEXES:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", " ", s)      # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace
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
    inter = len(a_set & b_set)
    uni = len(a_set | b_set)
    return inter / uni if uni else 0.0


def parse_created_series(s: pd.Series) -> pd.Series:
    """
    Try to parse created timestamps. Returns datetime64[ns] with NaT when unparseable.
    Accepts Jira-style and generic formats.
    """
    # pandas to_datetime is flexible; errors='coerce' makes NaT
    return pd.to_datetime(s, errors="coerce", utc=False)


def determine_keep_delete(i, j, created_dt: pd.Series | None):
    """
    Return (keep_idx, delete_idx) deterministically.
    If created_dt exists and both parse -> older KEEP.
    Else -> smaller index KEEP (earlier row).
    """
    if created_dt is not None:
        ci, cj = created_dt.iloc[i], created_dt.iloc[j]
        if pd.notna(ci) and pd.notna(cj):
            if ci <= cj:
                return i, j
            return j, i
    # fallback: earlier row KEEP
    return (i, j) if i < j else (j, i)


# ----------------------------
# UI
# ----------------------------
uploaded = st.file_uploader("Upload defects CSV", type=["csv"])

with st.expander("Active rules (click to view)", expanded=False):
    st.markdown(
        f"""
- Fields used: **Summary + Description**
- Ignore (inside text): version/build/device/log/repro/url/numbers
- EXACT: normalized text **%100 identical**
- SEMANTIC candidate: **shared tokens ≥ {MIN_SHARED_TOKENS}**
- SEMANTIC thresholds:
  - **Decision = SAFE_DELETE_STRICT** if similarity **≥ {THRESH_SAFE}**
  - **Decision = QA_REVIEW** if similarity **{THRESH_MIN}–{THRESH_SAFE - 0.01:.2f}**
- Output: **Single table** + summary
"""
    )

if not uploaded:
    st.stop()

# Read CSV robustly
raw = uploaded.getvalue().decode("utf-8", errors="replace")
df = pd.read_csv(StringIO(raw), sep=None, engine="python", on_bad_lines="skip")

cols = list(df.columns)

key_col = pick_col(cols, ["issue key", "issue_key", "key"]) or cols[0]
summary_col = pick_col(cols, ["summary", "title"]) or cols[0]
desc_col = pick_col(cols, ["description"]) or cols[0]
created_col = pick_col(cols, ["created", "created date", "created_at", "createdat"])

st.write("Detected columns:")
st.json(
    {
        "Issue Key": key_col,
        "Summary/Title": summary_col,
        "Description": desc_col,
        "Created (optional)": created_col if created_col else "(not found)",
    }
)

work = df.copy()
work["_row"] = np.arange(len(work))
work["_key"] = work[key_col].astype(str).str.strip()
work["_norm"] = [normalize_text(a, b) for a, b in zip(work[summary_col], work[desc_col])]
work["_tokens"] = work["_norm"].apply(tokenize)
work["_set"] = work["_tokens"].apply(set)

created_dt = None
if created_col:
    created_dt = parse_created_series(work[created_col])

# 1) EXACT (by normalized text identity)
seen_norm = {}
exact_pairs = []  # list of (keep_idx, delete_idx)
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

# 2) SEMANTIC (>= 0.69) using inverted index to avoid O(n^2)
inv = defaultdict(list)
for i, s in enumerate(work["_set"]):
    for t in s:
        inv[t].append(i)

semantic_pairs = []  # (keep_idx, delete_idx, score)
semantic_delete_idx = set()

seen_pair = set()

for i, si in enumerate(work["_set"]):
    if i in exact_delete_idx or i in semantic_delete_idx:
        continue

    # candidate generation
    cand = set()
    for t in si:
        cand.update(inv[t])

    for j in cand:
        if j == i:
            continue

        # avoid duplicates
        a, b = (j, i) if j < i else (i, j)
        if (a, b) in seen_pair:
            continue
        seen_pair.add((a, b))

        # compute
        sj = work["_set"].iloc[j]
        shared = len(si & sj)
        if shared < MIN_SHARED_TOKENS:
            continue

        score = jaccard(si, sj)
        if score < THRESH_MIN:
            continue

        keep_idx, del_idx = determine_keep_delete(i, j, created_dt)

        # do not mark something as delete twice
        if del_idx in exact_delete_idx or del_idx in semantic_delete_idx:
            continue

        semantic_pairs.append((keep_idx, del_idx, score))
        semantic_delete_idx.add(del_idx)

# Build combined output table
rows = []

for keep_idx, del_idx in exact_pairs:
    rows.append(
        {
            "Issue Key (Keep)": work["_key"].iloc[keep_idx],
            "Issue Key (Delete)": work["_key"].iloc[del_idx],
            "Type": "EXACT",
            "Similarity": 1.000,
            "Decision": "SAFE_DELETE_STRICT",
        }
    )

for keep_idx, del_idx, score in semantic_pairs:
    decision = "SAFE_DELETE_STRICT" if score >= THRESH_SAFE else "QA_REVIEW"
    rows.append(
        {
            "Issue Key (Keep)": work["_key"].iloc[keep_idx],
            "Issue Key (Delete)": work["_key"].iloc[del_idx],
            "Type": "SEMANTIC",
            "Similarity": round(float(score), 3),
            "Decision": decision,
        }
    )

out_df = pd.DataFrame(rows)

if not out_df.empty:
    # stable sort: SAFE_DELETE_STRICT first, then QA_REVIEW; then similarity desc
    order = {"SAFE_DELETE_STRICT": 0, "QA_REVIEW": 1}
    out_df["_ord"] = out_df["Decision"].map(order).fillna(9)
    out_df = out_df.sort_values(["_ord", "Similarity"], ascending=[True, False]).drop(columns=["_ord"])

# Summary
total = len(work)
exact_cnt = int((out_df["Type"] == "EXACT").sum()) if not out_df.empty else 0
semantic_cnt = int((out_df["Type"] == "SEMANTIC").sum()) if not out_df.empty else 0
safe_cnt = int((out_df["Decision"] == "SAFE_DELETE_STRICT").sum()) if not out_df.empty else 0
review_cnt = int((out_df["Decision"] == "QA_REVIEW").sum()) if not out_df.empty else 0

summary_df = pd.DataFrame(
    [
        {"Metric": "Total issue count", "Count": total},
        {"Metric": "Exact duplicate count", "Count": exact_cnt},
        {"Metric": "Semantic duplicate count (>=0.69)", "Count": semantic_cnt},
        {"Metric": "SAFE_DELETE_STRICT (>=0.80)", "Count": safe_cnt},
        {"Metric": "QA_REVIEW (0.69–0.79)", "Count": review_cnt},
        {"Metric": "Total flagged (safe+review)", "Count": len(out_df)},
    ]
)

# ----------------------------
# Render
# ----------------------------
st.subheader("Summary")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.subheader("Combined Output (SAFE_DELETE_STRICT + QA_REVIEW)")
st.dataframe(out_df, use_container_width=True, hide_index=True)

# Downloads
csv_out = out_df.to_csv(index=False).encode("utf-8")
csv_sum = summary_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Combined CSV",
    data=csv_out,
    file_name="BSDV_CLEAN_DEFECT_OUTPUT.csv",
    mime="text/csv",
)

st.download_button(
    "Download Summary CSV",
    data=csv_sum,
    file_name="BSDV_CLEAN_DEFECT_SUMMARY.csv",
    mime="text/csv",
)

st.caption("Note: This tool is conservative; QA_REVIEW items should be manually validated for same root cause.")
