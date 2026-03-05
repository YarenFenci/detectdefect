import re
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CANDIDATE_TFIDF_THRESHOLD = 0.35
TOP_K_NEIGHBORS           = 50
EMB_EXACT_THRESHOLD       = 0.97
EMB_SEMANTIC_THRESHOLD    = 0.88   # raised from 0.82 — tighter to reduce false positives
MIN_TOKENS                = 8

# Blending weights: problem signal dominates over full-text
FULL_TEXT_WEIGHT    = 0.40
PROBLEM_TEXT_WEIGHT = 0.60

SENTENCE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ──────────────────────────────────────────────
# Flow groups
# ──────────────────────────────────────────────
FLOW_GROUPS: Dict[str, Set[str]] = {
    "auth":         {"login","logout","signin","signup","register","otp","verification","verify",
                     "password","pin","biometrics","fingerprint","faceid","2fa","authentication"},
    "messaging":    {"message","chat","send","receive","delivery","delivered","read","unread",
                     "typing","sticker","emoji","gif","media","photo","video","file","document",
                     "forward","reply","delete","unsend","attachment"},
    "calling":      {"call","voice","videocall","video","ringing","ring","answer","decline",
                     "reject","missed","mute","speaker","bluetooth","headset","mic","microphone",
                     "echo","noise"},
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

# ──────────────────────────────────────────────
# Problem extraction
#
# The core insight from QA feedback:
#   "call log" appears in two different bugs —
#   one about wrong name, one about glare condition.
#   The system must focus on WHAT IS BROKEN, not WHERE.
#
# Strategy:
#   1. Split summary on separators (dash / pipe / colon)
#   2. Pick the segment with the most defect signal words
#   3. Strip leading context/location tokens
#   4. Append defect-signal sentences from description
# ──────────────────────────────────────────────

# Location / feature / screen context tokens — WHERE, not WHAT
CONTEXT_TOKENS: Set[str] = {
    "call", "log", "calllog", "history", "callhistory",
    "notification", "notifications", "screen", "page", "tab", "menu",
    "panel", "view", "list", "chat", "inbox", "feed", "home",
    "settings", "profile", "header", "footer", "toolbar", "bottom", "top",
    "briefing", "nowbriefing", "dialer", "keypad", "dialpad",
    "channel", "story", "status", "highlight",
    "search", "result", "results", "filter",
    "group", "contact", "contacts",
    "android", "ios", "iphone", "samsung", "device", "app", "application",
}

# Defect signal words — WHAT is broken
DEFECT_SIGNALS: List[str] = [
    "wrong", "incorrect", "invalid", "inaccurate", "mismatch", "mismatched",
    "misalign", "misaligned", "malformed",
    "missing", "disappear", "disappeared", "not shown", "not showing",
    "not display", "not displayed", "not visible", "hidden", "gone",
    "not appear", "not appearing",
    "not work", "not working", "broken", "fail", "failed", "fails",
    "unable", "cannot", "can't", "does not", "doesn't", "won't",
    "no longer", "stopped", "stop working",
    "crash", "crashes", "crashed", "freeze", "frozen", "hang", "hangs",
    "anr", "not responding", "force close", "black screen", "white screen",
    "duplicate", "duplicated", "duplicate entry",
    "not saved", "not updated", "not synced", "not cleared",
    "still showing", "showing old", "stale", "outdated",
    "unexpected", "unintended", "incorrect behavior", "wrong behavior",
    "created", "generated", "appearing", "showing",
    "even though", "despite", "although", "should not",
    "overlap", "overlapping", "truncated", "cut off", "clipped",
    "not centered", "layout issue",
    "receiving", "received", "not received", "still receiving",
    "incorrectly",
    "incorrect duration", "wrong duration", "incorrect time", "wrong time",
    "incorrect count", "wrong count", "incorrect number",
]

SEPARATOR_RE         = re.compile(r"\s*[-\u2013\u2014|:]\s*")
CONTEXT_PREP_RE      = re.compile(
    r"\s+(in|on|at|for|of|within|inside|under|from|during|after|when|while)\s+",
    re.IGNORECASE,
)


def _seg_defect_score(seg: str) -> int:
    sl = seg.lower()
    return sum(1 for sig in DEFECT_SIGNALS if sig in sl)


def extract_problem_text(summary, description) -> str:
    """
    Returns a problem-focused string stripped of location/context tokens.
    Used for problem embedding — the embedding sees WHAT is broken, not WHERE.
    """
    raw_summary = "" if pd.isna(summary) else str(summary).strip()
    raw_desc    = "" if pd.isna(description) else str(description).strip()

    # 1. Pick highest-signal segment from summary
    segments = SEPARATOR_RE.split(raw_summary)
    best_seg = raw_summary
    if len(segments) > 1:
        scored  = sorted(segments, key=_seg_defect_score, reverse=True)
        best_seg = scored[0].strip()
        if _seg_defect_score(best_seg) == 0:
            best_seg = raw_summary   # no clear winner, use full

    # 2. Strip trailing context after preposition
    prep_split = CONTEXT_PREP_RE.split(best_seg)
    if len(prep_split) >= 3:
        candidate = prep_split[0].strip()
        if _seg_defect_score(candidate) > 0:
            best_seg = candidate

    # 3. Remove pure context tokens word-by-word
    words         = best_seg.split()
    cleaned_words = [
        w for w in words
        if w.lower().rstrip("s") not in CONTEXT_TOKENS
        or any(sig in w.lower() for sig in DEFECT_SIGNALS)
    ]
    problem_core = " ".join(cleaned_words).strip()
    if len(problem_core.split()) < 3:
        problem_core = best_seg   # fallback: stripped too much

    # 4. Pull defect-signal sentences from description
    desc_sents   = re.split(r"[.\n]", raw_desc)
    defect_sents = []
    for sent in desc_sents[:10]:
        if any(sig in sent.lower() for sig in DEFECT_SIGNALS):
            clean = re.sub(r"\s+", " ", sent).strip()
            if len(clean) > 15:
                defect_sents.append(clean)
        if len(defect_sents) >= 2:
            break

    combined = problem_core
    if defect_sents:
        combined = problem_core + " " + " ".join(defect_sents)

    return combined.strip()


# ──────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────
def normalize_text(summary, desc) -> str:
    s = f"{'' if pd.isna(summary) else str(summary)} {'' if pd.isna(desc) else str(desc)}"
    s = s.lower()
    for pat in IGNORE_REGEXES:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_problem(raw: str) -> str:
    s = raw.lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\b\d+\.\d+\.\d+\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(norm: str) -> List[str]:
    return [t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS]


def get_flows(token_set: Set[str]) -> Set[str]:
    return {flow for flow, kws in FLOW_GROUPS.items() if token_set & kws}


def shared_flow_count(a: Set[str], b: Set[str]) -> int:
    return len(get_flows(a) & get_flows(b))


def parse_created(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def keep_delete_order(i: int, j: int, created_dt: Optional[pd.Series]) -> Tuple[int, int]:
    if created_dt is not None:
        ci, cj = created_dt.iloc[i], created_dt.iloc[j]
        if pd.notna(ci) and pd.notna(cj):
            return (i, j) if ci <= cj else (j, i)
    return (i, j) if i < j else (j, i)


def pick_col(cols: List[str], keywords: List[str]) -> Optional[str]:
    for c in cols:
        if any(k in c.lower() for k in keywords):
            return c
    return None


# ──────────────────────────────────────────────
# Sentence-transformers
# ──────────────────────────────────────────────
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
        list(texts),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


# ──────────────────────────────────────────────
# Data prep
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(raw_csv: str):
    df   = pd.read_csv(StringIO(raw_csv), sep=None, engine="python", on_bad_lines="skip")
    cols = list(df.columns)

    key_col     = pick_col(cols, ["issue key", "issue_key", "key"]) or cols[0]
    summary_col = pick_col(cols, ["summary", "title"])              or cols[0]
    desc_col    = pick_col(cols, ["description"])                   or cols[0]
    created_col = pick_col(cols, ["created", "created date", "created_at", "createdat"])
    tester_col  = pick_col(cols, ["assignee", "tester", "reporter", "assigned to", "owner"])

    w = df.copy()
    w["_key"]       = w[key_col].astype(str).str.strip()
    w["_norm"]      = [normalize_text(a, b) for a, b in zip(w[summary_col], w[desc_col])]
    w["_tokens"]    = w["_norm"].apply(tokenize)
    w["_set"]       = w["_tokens"].apply(set)
    w["_tok_count"] = w["_tokens"].apply(len)
    w["_raw"]       = [
        f"{'' if pd.isna(a) else str(a).strip()} {'' if pd.isna(b) else str(b).strip()}".strip()
        for a, b in zip(w[summary_col], w[desc_col])
    ]
    w["_problem"]   = [
        normalize_problem(extract_problem_text(a, b))
        for a, b in zip(w[summary_col], w[desc_col])
    ]
    w["_tester"]    = (
        w[tester_col].fillna("").astype(str).str.strip()
        if tester_col
        else pd.Series([""] * len(w))
    )

    created_dt = parse_created(w[created_col]) if created_col else None
    detected   = {
        "Issue Key":          key_col,
        "Summary / Title":    summary_col,
        "Description":        desc_col,
        "Created (optional)": created_col or "(not found)",
        "Tester / Assignee":  tester_col  or "(not found)",
    }
    return w, created_dt, detected


@st.cache_data(show_spinner="Building TF-IDF candidate index...")
def tfidf_neighbors(texts: List[str], topk: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    X  = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform(texts)
    nn = NearestNeighbors(
        n_neighbors=min(topk + 1, X.shape[0]),
        metric="cosine",
        algorithm="brute",
    ).fit(X)
    dist, idx = nn.kneighbors(X)
    return idx[:, 1:], 1.0 - dist[:, 1:]


# ──────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────
def run_pipeline(
    work: pd.DataFrame,
    created_dt: Optional[pd.Series],
    emb_full: np.ndarray,
    emb_problem: np.ndarray,
):
    n = len(work)

    # Blend: 60% problem-focused + 40% full-text, then re-normalize
    emb_blend = FULL_TEXT_WEIGHT * emb_full + PROBLEM_TEXT_WEIGHT * emb_problem
    norms     = np.linalg.norm(emb_blend, axis=1, keepdims=True)
    norms     = np.where(norms == 0, 1, norms)
    emb_blend = emb_blend / norms

    # Step 1 — TF-IDF speed gate
    nn_idx, nn_sim = tfidf_neighbors(work["_norm"].tolist(), TOP_K_NEIGHBORS)

    cand: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for pos in range(nn_idx.shape[1]):
            j = int(nn_idx[i, pos])
            c = float(nn_sim[i, pos])
            if c < CANDIDATE_TFIDF_THRESHOLD:
                continue
            a, b = (i, j) if i < j else (j, i)
            if a != b and c > cand.get((a, b), 0.0):
                cand[(a, b)] = c

    # Step 2 — blended embedding decision
    rows:    List[Dict] = []
    deleted: Set[int]   = set()

    for (a, b) in sorted(cand, key=lambda k: cand[k], reverse=True):
        ki, di = keep_delete_order(a, b, created_dt)
        if di in deleted:
            continue
        if work["_tok_count"].iloc[ki] < MIN_TOKENS or work["_tok_count"].iloc[di] < MIN_TOKENS:
            continue
        if shared_flow_count(work["_set"].iloc[ki], work["_set"].iloc[di]) < 1:
            continue

        score_blend   = float(np.dot(emb_blend[ki],   emb_blend[di]))
        score_problem = float(np.dot(emb_problem[ki], emb_problem[di]))

        if score_blend < EMB_SEMANTIC_THRESHOLD:
            continue

        set_ki    = work["_set"].iloc[ki]
        set_di    = work["_set"].iloc[di]
        tok_union = len(set_ki | set_di)
        jac       = len(set_ki & set_di) / tok_union if tok_union > 0 else 0.0

        dup_type = "Exact" if (
            work["_norm"].iloc[ki] == work["_norm"].iloc[di] or jac >= 0.90
        ) else "Semantic"

        rows.append({
            "Issue (Keep)":       work["_key"].iloc[ki],
            "Tester (Keep)":      work["_tester"].iloc[ki],
            "Issue (Duplicate)":  work["_key"].iloc[di],
            "Tester (Duplicate)": work["_tester"].iloc[di],
            "Type":               dup_type,
            "Score (Blend)":      round(score_blend,   3),
            "Score (Problem)":    round(score_problem, 3),
        })
        deleted.add(di)

    dup_df = (
        pd.DataFrame(rows)
        .sort_values("Score (Blend)", ascending=False)
        .reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=[
            "Issue (Keep)", "Tester (Keep)", "Issue (Duplicate)",
            "Tester (Duplicate)", "Type", "Score (Blend)", "Score (Problem)",
        ])
    )

    summary_df = pd.DataFrame([
        {"Metric": "Total issues",       "Value": n},
        {"Metric": "Semantic model",     "Value": SENTENCE_MODEL_NAME},
        {"Metric": "Semantic threshold", "Value": EMB_SEMANTIC_THRESHOLD},
        {"Metric": "Embedding blend",    "Value": f"{int(PROBLEM_TEXT_WEIGHT*100)}% problem + {int(FULL_TEXT_WEIGHT*100)}% full text"},
        {"Metric": "Duplicates found",   "Value": len(dup_df)},
        {"Metric": "Exact",              "Value": int((dup_df["Type"] == "Exact").sum())    if not dup_df.empty else 0},
        {"Metric": "Semantic",           "Value": int((dup_df["Type"] == "Semantic").sum()) if not dup_df.empty else 0},
    ])

    return summary_df, dup_df


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Defect Duplicate Detector", layout="wide")
    st.title("Defect Duplicate Detector")

    with st.sidebar:
        st.header("How it works")
        st.markdown(f"""
**Problem extraction**
Strips feature/screen context from summary, focuses on *what is broken*.

*Example:*
- `call log — wrong name shown` → **"wrong name shown"**
- `call log — duplicate entry after glare` → **"duplicate entry after glare"**
→ Correctly seen as **different bugs** despite same "call log" context.

**Dual embedding**
- {int(FULL_TEXT_WEIGHT*100)}% full text
- {int(PROBLEM_TEXT_WEIGHT*100)}% problem-focused text

**Threshold:** `{EMB_SEMANTIC_THRESHOLD}` (raised from 0.82)

**Score columns:**
- *Score (Blend)* — decision score
- *Score (Problem)* — problem-only similarity for reference
        """)

    model = load_sentence_model()
    if model is None:
        st.stop()

    uploaded = st.file_uploader("Upload defects CSV", type=["csv"], key="defect_csv_upload")
    if not uploaded:
        st.stop()

    raw = uploaded.getvalue().decode("utf-8", errors="replace")
    work, created_dt, detected = load_and_preprocess(raw)

    with st.expander("Detected columns", expanded=False):
        st.json(detected)

    # QA validation: show what the extractor pulled out
    with st.expander("Problem extraction preview — first 10 rows", expanded=False):
        st.caption("Check that 'Extracted Problem' captures the actual defect, not the screen name.")
        preview = work[["_key", "_raw", "_problem"]].head(10).copy()
        preview.columns = ["Issue Key", "Original Text", "Extracted Problem"]
        st.dataframe(preview, use_container_width=True, hide_index=True)

    emb_full = encode_texts(model, tuple(work["_raw"].tolist()))
    if emb_full is None:
        st.error("Full-text embedding failed.")
        st.stop()

    emb_problem = encode_texts(model, tuple(work["_problem"].tolist()))
    if emb_problem is None:
        st.error("Problem embedding failed.")
        st.stop()

    with st.spinner("Analysing…"):
        summary_df, dup_df = run_pipeline(work, created_dt, emb_full, emb_problem)

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Duplicate Issues")
    if dup_df.empty:
        st.info("No duplicates found above the confidence threshold.")
    else:
        st.dataframe(dup_df, use_container_width=True, hide_index=True)

    if not dup_df.empty:
        st.divider()
        dl_cols = [
            "Issue (Keep)", "Tester (Keep)",
            "Issue (Duplicate)", "Tester (Duplicate)",
            "Type", "Score (Blend)", "Score (Problem)",
        ]
        st.download_button(
            label="Download Results",
            data=dup_df[dl_cols].to_csv(index=False).encode("utf-8"),
            file_name="defect_duplicates.csv",
            mime="text/csv",
            key="dl_results",
        )


if __name__ == "__main__":
    main()
