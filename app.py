import re
from collections import deque
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CANDIDATE_TFIDF_THRESHOLD = 0.40
TOP_K_NEIGHBORS           = 50
EMB_EXACT_THRESHOLD       = 0.97   # cosine >= this -> Exact (near-identical content)
EMB_SEMANTIC_THRESHOLD    = 0.75   # cosine >= this -> Semantic (same meaning, different words)
MIN_TOKENS                = 8

SENTENCE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

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
# Helpers
# ──────────────────────────────────────────────
def pick_col(cols: List[str], keywords: List[str]) -> Optional[str]:
    for c in cols:
        if any(k in c.lower() for k in keywords):
            return c
    return None


def normalize_text(summary, desc) -> str:
    s = f"{'' if pd.isna(summary) else str(summary)} {'' if pd.isna(desc) else str(desc)}"
    s = s.lower()
    for pat in IGNORE_REGEXES:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
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


# ──────────────────────────────────────────────
# Data prep
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(raw_csv: str):
    df = pd.read_csv(StringIO(raw_csv), sep=None, engine="python", on_bad_lines="skip")
    cols = list(df.columns)

    key_col     = pick_col(cols, ["issue key", "issue_key", "key"]) or cols[0]
    summary_col = pick_col(cols, ["summary", "title"])              or cols[0]
    desc_col    = pick_col(cols, ["description"])                   or cols[0]
    created_col = pick_col(cols, ["created", "created date", "created_at", "createdat"])

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

    created_dt = parse_created(w[created_col]) if created_col else None
    detected = {
        "Issue Key":          key_col,
        "Summary / Title":    summary_col,
        "Description":        desc_col,
        "Created (optional)": created_col or "(not found)",
    }
    return w, created_dt, detected


@st.cache_data(show_spinner="Building TF-IDF candidate index...")
def tfidf_neighbors(texts: List[str], topk: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    X = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform(texts)
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
    emb: np.ndarray,
):
    n = len(work)

    # Step 1 — TF-IDF broad candidate generation (speed only)
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

    # Step 2 — embedding cosine is the only decision signal
    rows:    List[Dict]            = []
    deleted: Set[int]              = set()

    for (a, b) in sorted(cand, key=lambda k: cand[k], reverse=True):
        ki, di = keep_delete_order(a, b, created_dt)
        if di in deleted:
            continue

        if work["_tok_count"].iloc[ki] < MIN_TOKENS or work["_tok_count"].iloc[di] < MIN_TOKENS:
            continue

        if shared_flow_count(work["_set"].iloc[ki], work["_set"].iloc[di]) < 1:
            continue

        score = float(np.dot(emb[ki], emb[di]))
        if score < EMB_SEMANTIC_THRESHOLD:
            continue

        norm_ki = work["_norm"].iloc[ki]
        norm_di = work["_norm"].iloc[di]
        if norm_ki == norm_di or score >= EMB_EXACT_THRESHOLD:
            dup_type = "Exact"
        else:
            dup_type = "Semantic"

        rows.append({
            "Issue (Keep)":      work["_key"].iloc[ki],
            "Issue (Duplicate)": work["_key"].iloc[di],
            "Type":              dup_type,
            "Similarity":        round(score, 3),
        })
        deleted.add(di)

    dup_df = (
        pd.DataFrame(rows)
        .sort_values("Similarity", ascending=False)
        .reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["Issue (Keep)", "Issue (Duplicate)", "Type", "Similarity"])
    )

    summary_df = pd.DataFrame([
        {"Metric": "Total issues",          "Value": n},
        {"Metric": "Semantic model",        "Value": SENTENCE_MODEL_NAME},
        {"Metric": "Exact threshold",       "Value": EMB_EXACT_THRESHOLD},
        {"Metric": "Semantic threshold",    "Value": EMB_SEMANTIC_THRESHOLD},
        {"Metric": "Duplicates found",      "Value": len(dup_df)},
        {"Metric": "Exact",                 "Value": int((dup_df["Type"] == "Exact").sum())    if not dup_df.empty else 0},
        {"Metric": "Semantic",              "Value": int((dup_df["Type"] == "Semantic").sum()) if not dup_df.empty else 0},
    ])

    return summary_df, dup_df


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Defect Duplicate Detector", layout="wide")
    st.title("Defect Duplicate Detector")

    model = load_sentence_model()
    if model is None:
        st.stop()

    uploaded = st.file_uploader("Upload defects CSV", type=["csv"])
    if not uploaded:
        st.stop()

    raw = uploaded.getvalue().decode("utf-8", errors="replace")
    work, created_dt, detected = load_and_preprocess(raw)

    with st.expander("Detected columns", expanded=False):
        st.json(detected)

    emb = encode_texts(model, tuple(work["_raw"].tolist()))
    if emb is None:
        st.error("Embedding failed.")
        st.stop()

    with st.spinner("Analysing..."):
        summary_df, dup_df = run_pipeline(work, created_dt, emb)

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Duplicate Issues")
    if dup_df.empty:
        st.info("No duplicates found above the confidence threshold.")
    else:
        st.dataframe(dup_df, use_container_width=True, hide_index=True)

    st.divider()
    st.download_button(
        label="Download Results",
        data=dup_df[["Issue (Keep)", "Issue (Duplicate)", "Type"]].to_csv(index=False).encode("utf-8"),
        file_name="defect_duplicates.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
