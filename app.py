# app.py
import re
import io
import csv
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st


# ----------------------------
# BSDV_CLEAN_DEFECT - LOCKED RULES
# ----------------------------

# Minimal TR/EN stopwords (safe, deterministic). You can extend if you want.
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

    # logs
    r"\blogs?\s*[:=]\s*[^\n\r]+",
    r"\blogcat\s*[:=]\s*[^\n\r]+",
    r"\bstacktrace\s*[:=]\s*[\s\S]*?(?=\n{2,}|\Z)",

    # reproduction / steps blocks (keep conservative: remove headers + following lines until blank)
    r"\bsteps?\s*to\s*reproduce\b[\s\S]*?(?=\n{2,}|\Z)",
    r"\breproduction\b[\s\S]*?(?=\n{2,}|\Z)",
    r"\brepro\b[\s\S]*?(?=\n{2,}|\Z)",

    # device-ish lines (generic)
    r"\bdevice\s*[:=]\s*[^\n\r]+",
    r"\bmodel\s*[:=]\s*[^\n\r]+",
    r"\bandroid\s*(version)?\s*[:=]\s*[^\n\r]+",
    r"\bios\s*(version)?\s*[:=]\s*[^\n\r]+",

    # URLs
    r"https?://\S+",
    r"www\.\S+",

    # numbers (IDs, counts) - remove standalone numbers to avoid false non-duplicates
    r"\b\d+\b",
]


def normalize_text(s: str) -> str:
    """Normalize text by removing ignored sections and doing stable cleanup."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""

    s = str(s).lower()

    # Remove ignored patterns
    for pat in IGNORE_REGEXES:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)

    # Remove non-letter/digit separators -> keep spaces
    s = re.sub(r"[_/\\\-–—]", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    """Tokenize normalized text and drop stopwords and very short tokens."""
    s = normalize_text(s)
    toks = [t for t in s.split(" ") if t and t not in STOPWORDS and len(t) >= 3]
    return toks


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@dataclass(frozen=True)
class DuplicateRow:
    delete_key: str
    keep_key: str
    dup_type: str  # EXACT / SEMANTIC
    safe_delete: str = "YES"


def detect_duplicates(
    df: pd.DataFrame,
    col_key: str,
    col_summary: str,
    col_desc: str,
    semantic_jaccard_threshold: float = 0.78,
    semantic_seq_threshold: float = 0.78,
    min_shared_tokens: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Deterministic duplicate detection:
    - EXACT: normalized(summary+desc) identical.
    - SEMANTIC: not exact; Jaccard(tokens) >= threshold AND SequenceMatcher(norm_text) >= threshold
      AND shared tokens >= min_shared_tokens.
    Keep policy:
    - Keep the FIRST occurrence (row order) as canonical "Keep".
    """
    # Build normalized full text
    full_text = (df[col_summary].fillna("") + " " + df[col_desc].fillna("")).astype(str)
    norm_full = full_text.map(normalize_text)
    toks_full = full_text.map(tokenize)

    # EXACT grouping
    canonical_for_norm: Dict[str, str] = {}
    exact_dups: List[DuplicateRow] = []

    for i in range(len(df)):
        issue = str(df.iloc[i][col_key]).strip()
        nf = norm_full.iloc[i]
        if nf in canonical_for_norm:
            exact_dups.append(DuplicateRow(delete_key=issue, keep_key=canonical_for_norm[nf], dup_type="EXACT"))
        else:
            canonical_for_norm[nf] = issue

    # SEMANTIC (pairwise on remaining)
    # To keep it deterministic and not O(n^2) too heavy, we do blocking by first 3 tokens.
    # This is stable and speeds up for big files.
    index_by_block: Dict[str, List[int]] = {}
    for i in range(len(df)):
        toks = toks_full.iloc[i]
        block = " ".join(toks[:3]) if len(toks) >= 3 else "___"
        index_by_block.setdefault(block, []).append(i)

    exact_delete_set = set(r.delete_key for r in exact_dups)

    semantic_dups: List[DuplicateRow] = []
    used_as_delete = set()  # avoid multiple mappings for same delete item deterministically

    # For deterministic "Keep", we always map later row -> earliest matching keep in row order.
    for block, idxs in index_by_block.items():
        if len(idxs) < 2:
            continue

        idxs_sorted = sorted(idxs)
        for pos in range(1, len(idxs_sorted)):
            i = idxs_sorted[pos]
            issue_i = str(df.iloc[i][col_key]).strip()
            if issue_i in exact_delete_set or issue_i in used_as_delete:
                continue

            best_keep: Optional[int] = None
            best_score: float = 0.0

            ni = norm_full.iloc[i]
            ti = toks_full.iloc[i]

            for j in idxs_sorted[:pos]:
                issue_j = str(df.iloc[j][col_key]).strip()
                if issue_j in exact_delete_set:
                    continue

                nj = norm_full.iloc[j]
                tj = toks_full.iloc[j]

                jac = jaccard(ti, tj)
                if jac < semantic_jaccard_threshold:
                    continue

                sr = seq_ratio(ni, nj)
                if sr < semantic_seq_threshold:
                    continue

                shared = len(set(ti) & set(tj))
                if shared < min_shared_tokens:
                    continue

                score = 0.5 * jac + 0.5 * sr
                if score > best_score:
                    best_score = score
                    best_keep = j

            if best_keep is not None:
                keep_key = str(df.iloc[best_keep][col_key]).strip()
                semantic_dups.append(DuplicateRow(delete_key=issue_i, keep_key=keep_key, dup_type="SEMANTIC"))
                used_as_delete.add(issue_i)

    safe_delete_rows = exact_dups + semantic_dups

    safe_delete_df = pd.DataFrame(
        [
            {
                "Issue Key (Delete)": r.delete_key,
                "Duplicate Of (Keep)": r.keep_key,
                "Type": r.dup_type,
                "Safe Delete": r.safe_delete,
            }
            for r in safe_delete_rows
        ]
    )

    summary = {
        "Total defect count": int(len(df)),
        "Exact duplicate count": int(len(exact_dups)),
        "Semantic duplicate count": int(len(semantic_dups)),
        "Safe delete total": int(len(safe_delete_rows)),
    }
    return safe_delete_df, summary


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="BSDV_CLEAN_DEFECT", layout="wide")
st.title("BSDV_CLEAN_DEFECT — Defect Duplicate Cleaner (Summary + Description)")

with st.expander("Rules (Locked)", expanded=False):
    st.markdown(
        """
- **Fields used:** Summary, Description  
- **Ignored differences:** log / version / build / reproduction steps / device lines / URLs / numbers  
- **Decision:**
  - **EXACT:** normalized text identical → Safe Delete YES
  - **SEMANTIC:** same bug intent (deterministic similarity rules) → Safe Delete YES
  - Otherwise: NOT DUPLICATE (not listed)
- **Output:** SAFE_DELETE_DEFECT.csv + summary table
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

st.subheader("Semantic Thresholds (Deterministic)")
t1, t2, t3 = st.columns(3)
with t1:
    sem_jacc = st.slider("Semantic Jaccard threshold", 0.50, 0.95, 0.78, 0.01)
with t2:
    sem_seq = st.slider("Semantic Sequence threshold", 0.50, 0.95, 0.78, 0.01)
with t3:
    min_shared = st.number_input("Min shared tokens", 1, 20, 5, 1)

run = st.button("Run BSDV_CLEAN_DEFECT", type="primary")

if run:
    # Basic validation
    for required in [col_key, col_summary, col_desc]:
        if required not in df.columns:
            st.error(f"Missing column: {required}")
            st.stop()

    safe_delete_df, summary = detect_duplicates(
        df=df,
        col_key=col_key,
        col_summary=col_summary,
        col_desc=col_desc,
        semantic_jaccard_threshold=float(sem_jacc),
        semantic_seq_threshold=float(sem_seq),
        min_shared_tokens=int(min_shared),
    )

    st.success("Analysis complete.")

    # Summary table
    st.subheader("Summary")
    summary_df = pd.DataFrame([summary])
    st.dataframe(summary_df, use_container_width=True)

    # SAFE_DELETE output
    st.subheader("SAFE_DELETE_DEFECT.csv")
    st.dataframe(safe_delete_df, use_container_width=True)

    st.download_button(
        label="Download SAFE_DELETE_DEFECT.csv",
        data=to_csv_bytes(safe_delete_df),
        file_name="SAFE_DELETE_DEFECT.csv",
        mime="text/csv",
    )
