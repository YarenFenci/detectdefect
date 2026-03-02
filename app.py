
import re
from collections import defaultdict
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="BSDV_CLEAN_DEFECT", layout="wide")
st.title("BSDV_CLEAN_DEFECT (V2)")
st.caption("Single-table output: SAFE_DELETE_STRICT + QA_REVIEW, with Duplicate Type column.")


# ----------------------------
# Locked config (your standard)
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

IGNORE_REGEXES = [
    r"\b(app\s*)?version\s*[:=]\s*[^\n\r]+",
    r"\bbuild\s*[:=]\s*[^\n\r]+",
    r"\bdevice\s*[:=]\s*[^\n\r]+",
    r"\blogs?\s*[:=]\s*[^\n\r]+",
    r"\brepro(duction)?\b.*",  # if repro text is embedded, ignore
    r"https?://\S+",
    r"\b\d+\.\d+\.\d+(\.\d+)?\b",
    r"\b\d+\b",
]


# ----------------------------
# Helpers
# ----------------------------
def pick_col(cols, must_contain_any):
    """Pick first column whose lowercase name contains any keyword."""
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


def parse_created(series: pd.Series) -> pd.Series:
    """Parse created timestamps. Unparseable -> NaT."""
    return pd.to_datetime(series, errors="coerce", utc=False)


def determine_keep_delete(i, j, created_dt: pd.Series | None):
    """
    Deterministic KEEP rule:
      - if created_dt exists and both parseable -> older KEEP
      - else -> smaller index KEEP (earlier row)
    """
    if created_dt is not None:
        ci, cj = created_dt.iloc[i], created_dt.iloc[j]
        if pd.notna(ci) and pd.notna(cj):
            return (i, j) if ci <= cj else (j, i)
    return (i, j) if i < j else (j, i)


# ----------------------------
# Upload
# ----------------------------
uploaded = st.file_uploader("Upload defects CSV", type=["csv"])
with st.expander("Active rules", expanded=False):
    st.markdown(
        f"""
**Fields used:** Summary + Description  
**SEMANTIC thresholds:** shared tokens ≥ {MIN_SHARED_TOKENS}, similarity ≥ {THRESH_MIN}  
- **SAFE_DELETE_STRICT** if similarity ≥ {THRESH_SAFE}  
- **QA_REVIEW** if {THRESH_MIN} ≤ similarity < {THRESH_SAFE}  
"""
    )

if not uploaded:
    st.stop()

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

# ----------------------------
# Prepare work df
# ----------------------------
work = df.copy()
work["_row"] = np.arange(len(work))
work["_key"] = work[key_col].astype(str).str.strip()
work["_norm"] = [normalize_text(a, b) for a, b in zip(work[summary_col], work[desc_col])]
work["_set"] = work["_norm"].apply(lambda x: set(tokenize(x)))

created_dt = None
if created_col:
    created_dt = parse_created(work[created_col])

# ----------------------------
# 1) EXACT duplicates
# ----------------------------
seen_norm = {}
exact_pairs = []          # (keep_idx, delete_idx)
exact_delete_idx = set()  # delete indices to exclude from semantic

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
# 2) SEMANTIC duplicates (candidate index)
# ----------------------------
inv = defaultdict(list)
for i, s in enumerate(work["_set"]):
    for t in s:
        inv[t].append(i)

semantic_pairs = []          # (keep_idx, delete_idx, score)
semantic_delete_idx = set()  # ensure one delete target once
seen_pair = set()            # avoid duplicate pair scoring

for i, si in enumerate(work["_set"]):
    if i in exact_delete_idx or i in semantic_delete_idx:
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
        shared = len(si & sj)
        if shared < MIN_SHARED_TOKENS:
            continue

        score = jaccard(si, sj)
        if score < THRESH_MIN:
            continue

        keep_idx, del_idx = determine_keep_delete(i, j, created_dt)
        if del_idx in exact_delete_idx or del_idx in semantic_delete_idx:
            continue

        semantic_pairs.append((keep_idx, del_idx, score))
        semantic_delete_idx.add(del_idx)

# ----------------------------
# Build single combined output
# ----------------------------
rows = []

# EXACT -> always SAFE_DELETE_STRICT
for keep_idx, del_idx in exact_pairs:
    rows.append(
        {
            "Issue Key (Keep)": work["_key"].iloc[keep_idx],
            "Issue Key (Delete)": work["_key"].iloc[del_idx],
            "Duplicate Type": "EXACT",
            "Similarity": 1.000,
            "Decision": "SAFE_DELETE_STRICT",
        }
    )

# SEMANTIC -> decision by score
for keep_idx, del_idx, score in semantic_pairs:
    decision = "SAFE_DELETE_STRICT" if score >= THRESH_SAFE else "QA_REVIEW"
    rows.append(
        {
            "Issue Key (Keep)": work["_key"].iloc[keep_idx],
            "Issue Key (Delete)": work["_key"].iloc[del_idx],
            "Duplicate Type": "SEMANTIC",
            "Similarity": round(float(score), 3),
            "Decision": decision,
        }
    )

out_df = pd.DataFrame(rows)

# stable sort: SAFE_DELETE_STRICT first, then QA_REVIEW; then similarity desc
if not out_df.empty:
    decision_order = {"SAFE_DELETE_STRICT": 0, "QA_REVIEW": 1}
    out_df["_ord"] = out_df["Decision"].map(decision_order).fillna(9)
    out_df = out_df.sort_values(["_ord", "Similarity"], ascending=[True, False]).drop(columns=["_ord"])

# ----------------------------
# Summary table
# ----------------------------
total = len(work)
exact_cnt = int((out_df["Duplicate Type"] == "EXACT").sum()) if not out_df.empty else 0
semantic_cnt = int((out_df["Duplicate Type"] == "SEMANTIC").sum()) if not out_df.empty else 0
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

# ----------------------------
# Downloads
# ----------------------------
st.download_button(
    "Download Combined CSV",
    data=out_df.to_csv(index=False).encode("utf-8"),
    file_name="BSDV_CLEAN_DEFECT_OUTPUT.csv",
    mime="text/csv",
)

st.download_button(
    "Download Summary CSV",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name="BSDV_CLEAN_DEFECT_SUMMARY.csv",
    mime="text/csv",
)

st.caption("QA_REVIEW items must be validated for same root cause before deletion.")
