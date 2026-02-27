# app.py
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import streamlit as st


# =========================
# BSDV_CLEAN_DEFECT (LOCKED)
# =========================
# Fields used: Summary + Description
# Ignore: log, version, build, reproduction/steps, device lines, URLs, numbers
# Decision:
#   EXACT    -> normalized(full_text) identical
#   SEMANTIC -> same bug intent after ignores (deterministic token similarity)
#
# Output:
#   - Summary counts (Total / Exact / Semantic / Safe delete)
#   - SAFE delete list table + CSV download
#
# Determinism:
#   - Keep = earliest row in file order
#   - For each delete, pick the earliest matching keep


STOPWORDS = set("""
a an the and or but if then else when while for to of in on at by with without from into
is are was were be been being this that these those it its as
ve veya ama eğer ise değil için ile
""".split())

IGNORE_REGEXES = [
    # versions / builds
    r"\b(app\s*)?version\s*[:=]\s*[^\n\r]+",
    r"\bbuild\s*[:=]\s*[^\n\r]+",
    r"\bver\s*[:=]\s*[^\n\r]+",
    r"\b\d+\.\d+\.\d+(\.\d+)?\b",

    # logs / stack traces
    r"\blogs?\s*[:=]\s*[^\n\r]+",
    r"\blogcat\s*[:=]\s*[^\n\r]+",
    r"\bstack\s*trace\b[\s\S]*?(?=\n{2,}|\Z)",
    r"\bstacktrace\b[\s\S]*?(?=\n{2,}|\Z)",

    # reproduction / steps
    r"\bsteps?\s*to\s*reproduce\b[\s\S]*?(?=\n{2,}|\Z)",
    r"\breproduction\b[\s\S]*?(?=\n{2,}|\Z)",
    r"\brepro\b[\s\S]*?(?=\n{2,}|\Z)",

    # device-ish lines
    r"\bdevice\s*[:=]\s*[^\n\r]+",
    r"\bmodel\s*[:=]\s*[^\n\r]+",
    r"\bandroid\s*(version)?\s*[:=]\s*[^\n\r]+",
    r"\bios\s*(version)?\s*[:=]\s*[^\n\r]+",

    # URLs
    r"https?://\S+",
    r"www\.\S+",

    # standalone numbers
    r"\b\d+\b",
]


def normalize_text(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).lower()

    for pat in IGNORE_REGEXES:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)

    s = re.sub(r"[_/\\\-–—]", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(norm: str) -> List[str]:
    # drop stopwords + short tokens
    toks = [t for t in norm.split() if t and len(t) >= 3 and t not in STOPWORDS]
    return toks


def jaccard_set(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


@dataclass(frozen=True)
class Dup:
    delete_key: str
    keep_key: str
    dup_type: str  # EXACT / SEMANTIC
    safe_delete: str = "YES"


def detect_bsdv_clean_defect(
    df: pd.DataFrame,
    key_col: str,
    summary_col: str,
    desc_col: str,
    # LOCKED defaults (change here once, not via UI, if you want)
    semantic_jaccard_threshold: float = 0.80,   # aligns with “strong semantic”
    min_shared_tokens: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    # Build normalized full text
    full = (df[summary_col].fillna("") + " " + df[desc_col].fillna("")).astype(str)
    norm = full.map(normalize_text)

    issue_keys = df[key_col].astype(str).map(lambda x: x.strip())

    # EXACT duplicates: identical normalized text
    canonical_for_norm: Dict[str, str] = {}
    exact_dups: List[Dup] = []
    exact_delete_set: Set[str] = set()

    for i in range(len(df)):
        ik = issue_keys.iloc[i]
        nf = norm.iloc[i]
        if nf in canonical_for_norm:
            exact_dups.append(Dup(delete_key=ik, keep_key=canonical_for_norm[nf], dup_type="EXACT"))
            exact_delete_set.add(ik)
        else:
            canonical_for_norm[nf] = ik

    # SEMANTIC duplicates (deterministic):
    # We do candidate generation by token index to avoid O(n^2) blowup.
    tokens_list = norm.map(tokenize)
    token_sets = tokens_list.map(set)

    # Build inverted index: token -> list of row indices (only for tokens that appear not too frequently)
    token_freq: Dict[str, int] = {}
    for ts in token_sets:
        for t in ts:
            token_freq[t] = token_freq.get(t, 0) + 1

    # Keep "useful" tokens (rare-ish) for blocking
    # deterministic: threshold based on dataset size
    n = len(df)
    max_freq = max(5, int(n * 0.10))  # tokens appearing in >10% rows are too common
    useful_tokens = {t for t, f in token_freq.items() if f <= max_freq}

    inv: Dict[str, List[int]] = {}
    for i in range(n):
        for t in (token_sets.iloc[i] & useful_tokens):
            inv.setdefault(t, []).append(i)

    semantic_dups: List[Dup] = []
    semantic_delete_set: Set[str] = set()

    for i in range(n):
        del_key = issue_keys.iloc[i]
        if del_key in exact_delete_set:
            continue

        # gather candidates from shared useful tokens (only earlier rows as KEEP)
        cand: Set[int] = set()
        for t in (token_sets.iloc[i] & useful_tokens):
            for j in inv.get(t, []):
                if j < i:
                    cand.add(j)

        if not cand:
            continue

        best_keep_idx: Optional[int] = None
        best_score = 0.0

        si = token_sets.iloc[i]

        # deterministic: evaluate candidates in ascending index
        for j in sorted(cand):
            keep_key = issue_keys.iloc[j]
            if keep_key in exact_delete_set:
                continue

            sj = token_sets.iloc[j]

            shared = len(si & sj)
            if shared < min_shared_tokens:
                continue

            jac = jaccard_set(si, sj)
            if jac < semantic_jaccard_threshold:
                continue

            # pick best score; tie-breaker = earlier keep (already ensured by j order)
            if jac > best_score:
                best_score = jac
                best_keep_idx = j

        if best_keep_idx is not None and del_key not in semantic_delete_set:
            semantic_dups.append(Dup(delete_key=del_key, keep_key=issue_keys.iloc[best_keep_idx], dup_type="SEMANTIC"))
            semantic_delete_set.add(del_key)

    safe_rows = exact_dups + semantic_dups

    safe_df = pd.DataFrame(
        [{
            "Issue Key (Delete)": d.delete_key,
            "Duplicate Of (Keep)": d.keep_key,
            "Type": d.dup_type,
            "Safe Delete": d.safe_delete,
        } for d in safe_rows]
    )

    # Counts
    exact_count = len(exact_dups)
    semantic_count = len(semantic_dups)

    summary = {
        "Total issue count": int(n),
        "Exact duplicate count": int(exact_count),
        "Semantic duplicate count": int(semantic_count),
        "Safe delete total": int(len(safe_rows)),
    }
    return safe_df, summary


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="BSDV_CLEAN_DEFECT", layout="wide")
st.title("BSDV_CLEAN_DEFECT — Defect Duplicate Cleaner (EXACT + SEMANTIC)")

with st.expander("Locked Rules (BSDV_CLEAN_DEFECT)", expanded=False):
    st.markdown(
        """
**Fields used:** Summary + Description  
**Ignored:** log / version / build / reproduction / device lines / URLs / numbers  
**Output:** Summary counts + SAFE Delete list (table) + CSV
        """
    )

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to run BSDV_CLEAN_DEFECT.")
    st.stop()

# CSV read options
c1, c2, c3, c4 = st.columns(4)
with c1:
    delimiter = st.selectbox("Delimiter", options=[",", ";", "\t", "|"], index=1)
with c2:
    encoding = st.selectbox("Encoding", options=["utf-8", "utf-8-sig", "latin-1"], index=0)
with c3:
    quotechar = st.selectbox("Quote char", options=['"', "'"], index=0)
with c4:
    preview_rows = st.number_input("Preview rows", min_value=5, max_value=200, value=20, step=5)

try:
    df = pd.read_csv(uploaded, sep=delimiter, encoding=encoding, quotechar=quotechar, engine="python")
except Exception as e:
    st.error(f"CSV read error: {e}")
    st.stop()

st.subheader("Input Preview")
st.dataframe(df.head(int(preview_rows)), use_container_width=True)

# Column mapping
st.subheader("Column Mapping")
cols = list(df.columns)
m1, m2, m3 = st.columns(3)
with m1:
    col_key = st.selectbox("Issue Key column", cols, index=cols.index("Issue key") if "Issue key" in cols else 0)
with m2:
    col_summary = st.selectbox("Summary column", cols, index=cols.index("Summary") if "Summary" in cols else 0)
with m3:
    col_desc = st.selectbox("Description column", cols, index=cols.index("Description") if "Description" in cols else 0)

run = st.button("Run BSDV_CLEAN_DEFECT", type="primary")

if run:
    safe_df, summary = detect_bsdv_clean_defect(
        df=df,
        key_col=col_key,
        summary_col=col_summary,
        desc_col=col_desc,
        # LOCKED params live in function defaults
    )

    # ---- Summary counts (DIRECT OUTPUT) ----
    st.subheader("Summary (Direct Counts)")
    # show as 4 metrics + a table
    a, b, c, d = st.columns(4)
    a.metric("Total issues", summary["Total issue count"])
    b.metric("EXACT", summary["Exact duplicate count"])
    c.metric("SEMANTIC", summary["Semantic duplicate count"])
    d.metric("SAFE DELETE", summary["Safe delete total"])
    st.table(pd.DataFrame([summary]))

    # ---- SAFE Delete table ----
    st.subheader("SAFE Delete List (Table)")
    if len(safe_df) == 0:
        st.write("No SAFE_DELETE items found.")
    else:
        # deterministic ordering: show EXACT first, then SEMANTIC; inside, by Keep then Delete
        safe_df_show = safe_df.copy()
        safe_df_show["TypeOrder"] = safe_df_show["Type"].map({"EXACT": 0, "SEMANTIC": 1}).fillna(9)
        safe_df_show = safe_df_show.sort_values(["TypeOrder", "Duplicate Of (Keep)", "Issue Key (Delete)"]).drop(columns=["TypeOrder"])
        st.table(safe_df_show)

        # Type breakdown (table)
        st.subheader("Breakdown by Type")
        breakdown = safe_df["Type"].value_counts().rename_axis("Type").reset_index(name="Count")
        st.table(breakdown)

    st.download_button(
        "Download SAFE_DELETE_DEFECT.csv",
        data=to_csv_bytes(safe_df),
        file_name="SAFE_DELETE_DEFECT.csv",
        mime="text/csv",
    )
