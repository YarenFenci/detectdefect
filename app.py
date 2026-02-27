# app.py
import re
import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import streamlit as st


# =========================
# BSDV_CLEAN_DEFECT (LOCKED)
# =========================
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
    return [t for t in norm.split(" ") if t and len(t) >= 3 and t not in STOPWORDS]


def jaccard_tokens(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def shingles(tokens: List[str], k: int = 3) -> List[str]:
    if len(tokens) < k:
        return [" ".join(tokens)] if tokens else [""]
    return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]


def band_keys(tokens: List[str]) -> Set[str]:
    sh = shingles(tokens, k=3)
    if not sh:
        return set()
    keys = set()
    for salt in ("A", "B", "C", "D", "E", "F"):
        m = None
        for x in sh:
            h = stable_hash(salt + "|" + x)
            m = h if m is None else min(m, h)
        keys.add(f"{salt}:{m}")
    return keys


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
    semantic_jaccard_threshold: float,
    semantic_seq_threshold: float,
    min_shared_tokens: int,
    max_candidates_per_row: int = 250,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    full = (df[summary_col].fillna("") + " " + df[desc_col].fillna("")).astype(str)
    norm = full.map(normalize_text)
    toks = norm.map(tokenize)

    issue_keys = df[key_col].astype(str).map(lambda x: x.strip())

    # EXACT
    canonical_for_norm: Dict[str, str] = {}
    exact_dups: List[Dup] = []
    exact_deletes: Set[str] = set()

    for i in range(len(df)):
        ik = issue_keys.iloc[i]
        nf = norm.iloc[i]
        if nf in canonical_for_norm:
            exact_dups.append(Dup(delete_key=ik, keep_key=canonical_for_norm[nf], dup_type="EXACT"))
            exact_deletes.add(ik)
        else:
            canonical_for_norm[nf] = ik

    # SEMANTIC: LSH-like candidate gen + scoring
    band_index: Dict[str, List[int]] = {}
    bands_per_row: List[Set[str]] = []

    for i in range(len(df)):
        bkeys = band_keys(toks.iloc[i])
        bands_per_row.append(bkeys)
        for bk in bkeys:
            band_index.setdefault(bk, []).append(i)

    semantic_dups: List[Dup] = []
    semantic_deletes: Set[str] = set()

    for i in range(len(df)):
        ik_i = issue_keys.iloc[i]
        if ik_i in exact_deletes:
            continue

        cand: Set[int] = set()
        for bk in bands_per_row[i]:
            for j in band_index.get(bk, []):
                if j < i:
                    cand.add(j)

        if not cand:
            continue

        cand_sorted = sorted(cand)
        if len(cand_sorted) > max_candidates_per_row:
            cand_sorted = cand_sorted[:max_candidates_per_row]

        best_j: Optional[int] = None
        best_score = 0.0

        ni = norm.iloc[i]
        ti = toks.iloc[i]

        for j in cand_sorted:
            ik_j = issue_keys.iloc[j]
            if ik_j in exact_deletes:
                continue

            nj = norm.iloc[j]
            tj = toks.iloc[j]

            shared = len(set(ti) & set(tj))
            if shared < min_shared_tokens:
                continue

            jac = jaccard_tokens(ti, tj)
            if jac < semantic_jaccard_threshold:
                continue

            sr = seq_ratio(ni, nj)
            if sr < semantic_seq_threshold:
                continue

            score = 0.5 * jac + 0.5 * sr
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            keep = issue_keys.iloc[best_j]
            if ik_i not in semantic_deletes:
                semantic_dups.append(Dup(delete_key=ik_i, keep_key=keep, dup_type="SEMANTIC"))
                semantic_deletes.add(ik_i)

    safe_rows = exact_dups + semantic_dups

    safe_delete_df = pd.DataFrame(
        [{
            "Issue Key (Delete)": d.delete_key,
            "Duplicate Of (Keep)": d.keep_key,
            "Type": d.dup_type,
            "Safe Delete": d.safe_delete,
        } for d in safe_rows]
    )

    summary = {
        "Total defect count": int(len(df)),
        "Exact duplicate count": int(len(exact_dups)),
        "Semantic duplicate count": int(len(semantic_dups)),
        "Safe delete total": int(len(safe_rows)),
    }
    return safe_delete_df, summary


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="BSDV_CLEAN_DEFECT", layout="wide")
st.title("BSDV_CLEAN_DEFECT — Defect Duplicate Cleaner (EXACT + SEMANTIC)")

with st.expander("Locked Rules", expanded=False):
    st.markdown(
        """
**Fields used:** Summary + Description  
**Ignored differences:** log / version / build / reproduction / device lines / URLs / numbers  
**Output:**
- Summary Table (tablo)
- SAFE Delete List (tablo) + CSV download
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

# Semantic settings (you can LOCK by removing sliders and hardcoding values)
st.subheader("Semantic Settings (Deterministic)")
t1, t2, t3 = st.columns(3)
with t1:
    sem_jacc = st.slider("Jaccard(token) threshold", 0.50, 0.95, 0.78, 0.01)
with t2:
    sem_seq = st.slider("SequenceMatcher threshold", 0.50, 0.95, 0.78, 0.01)
with t3:
    min_shared = st.number_input("Min shared tokens", 1, 30, 5, 1)

run = st.button("Run BSDV_CLEAN_DEFECT", type="primary")

if run:
    safe_df, summary = detect_bsdv_clean_defect(
        df=df,
        key_col=col_key,
        summary_col=col_summary,
        desc_col=col_desc,
        semantic_jaccard_threshold=float(sem_jacc),
        semantic_seq_threshold=float(sem_seq),
        min_shared_tokens=int(min_shared),
        max_candidates_per_row=250,
    )

    st.success("Analysis complete.")

    # ---- Summary TABLE (Tablo) ----
    st.subheader("Summary Table")
    summary_df = pd.DataFrame([summary])
    st.table(summary_df)  # table view (not interactive)

    # ---- SAFE DELETE LIST TABLE ----
    st.subheader("SAFE Delete List (Table)")

    show_exact = st.checkbox("Show EXACT", value=True)
    show_sem = st.checkbox("Show SEMANTIC", value=True)

    filtered = safe_df.copy()
    allowed = set()
    if show_exact:
        allowed.add("EXACT")
    if show_sem:
        allowed.add("SEMANTIC")
    if allowed:
        filtered = filtered[filtered["Type"].isin(list(allowed))]
    else:
        filtered = filtered.iloc[0:0]

    st.table(filtered)  # table view

    # Optional: counts by Type
    st.subheader("Type Breakdown")
    if len(safe_df) == 0:
        st.write("No SAFE_DELETE items found.")
    else:
        st.table(safe_df["Type"].value_counts().rename_axis("Type").reset_index(name="Count"))

    # Download
    st.download_button(
        label="Download SAFE_DELETE_DEFECT.csv",
        data=to_csv_bytes(safe_df),
        file_name="SAFE_DELETE_DEFECT.csv",
        mime="text/csv",
    )
