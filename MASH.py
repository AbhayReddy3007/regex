# app.py
# Streamlit app: Upload file -> 'sentences' (MASH + resolution) -> 'extracted values' (ONLY % numbers within ±5 words of MASH/resolution) -> drop empty -> download

import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MASH Resolution Extractor", layout="wide")
st.title("MASH + Resolution Sentence & % Number Extractor (±5 words)")
st.caption("Upload an Excel (.xlsx) or CSV with an **abstracts** column.")

uploaded = st.file_uploader("Upload file", type=["xlsx", "csv"])

# ---------------- helpers ----------------
SENT_JOINER = " || "
KEYWORDS = {"mash", "resolution"}  # anchors
WINDOW = 5                         # words on each side

def split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [text.strip()]

def build_sentences_column(series: pd.Series) -> pd.Series:
    sent_lists = series.apply(split_sentences)
    filtered = sent_lists.apply(
        lambda lst: [s for s in lst
                     if re.search(r'\bMASH\b', s, re.IGNORECASE)
                     and re.search(r'\bresolution\b', s, re.IGNORECASE)]
    )
    return filtered.apply(lambda lst: SENT_JOINER.join(lst))

def _strip_punct(word: str) -> str:
    return re.sub(r'^\W+|\W+$', '', word or '')

def _window_percent_numbers_from_sentence(sentence: str, window: int = WINDOW):
    """
    Return ONLY percent numbers within ±window words of any 'MASH' or 'resolution'.
    Accepts '12%', '12 %', '1,234.5 %', outputs canonical '12%' (no space).
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return []

    tokens = sentence.split()
    anchor_idxs = [
        i for i, tok in enumerate(tokens)
        if _strip_punct(tok).lower() in KEYWORDS
    ]
    if not anchor_idxs:
        return []

    hits, seen = [], set()
    # Require a percent sign (optionally spaced): group(2) exists and includes any spaces before %
    num_pct_re = re.compile(r'((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(\s*%)')  # % is REQUIRED

    for i in anchor_idxs:
        start = max(0, i - window)
        end = min(len(tokens), i + window + 1)
        window_text = " ".join(tokens[start:end])

        for m in num_pct_re.finditer(window_text):
            num = m.group(1).replace(',', '')
            val = num + '%'
            if val not in seen:
                seen.add(val)
                hits.append(val)

    return hits

def extract_windowed_percent_from_sentences_field(sentences_field: str):
    if not isinstance(sentences_field, str) or not sentences_field.strip():
        return []
    all_vals, seen = [], set()
    for sent in sentences_field.split(SENT_JOINER):
        vals = _window_percent_numbers_from_sentence(sent)
        for v in vals:
            if v not in seen:
                seen.add(v)
                all_vals.append(v)
    return all_vals

# ---------------- main ----------------
if uploaded is None:
    st.info("⬆️ Choose a file to begin.")
else:
    try:
        # Read file
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
            output_kind = "csv"
        else:
            df = pd.read_excel(uploaded)
            output_kind = "xlsx"

        st.write("**Columns detected:**", list(df.columns))

        if 'abstracts' not in df.columns:
            st.error("The file must contain a column named **abstracts**.")
            st.stop()

        df['abstracts'] = df['abstracts'].astype(str)

        # 1) sentences column (must contain BOTH 'MASH' and 'resolution')
        df['sentences'] = build_sentences_column(df['abstracts'])

        # 2) extracted values: ONLY % numbers within ±5 words of anchors
        df['extracted values'] = df['sentences'].apply(
            lambda s: ", ".join(extract_windowed_percent_from_sentences_field(s)) if isinstance(s, str) else ""
        )

        # 3) drop rows where 'sentences' is empty
        has_sentences = df['sentences'].astype(str).str.strip().astype(bool)
        df = df[has_sentences].copy()

        # 4) drop rows where 'extracted values' is empty
        has_values = df['extracted values'].astype(str).str.strip().astype(bool)
        df_out = df[has_values].copy()

        if df_out.empty:
            st.warning(
                "No rows left after filtering. "
                "Either there are no sentences containing both 'MASH' and 'resolution', "
                "or there are no percent values within ±5 words of those keywords."
            )
        else:
            st.success(f"Processed {len(df_out)} rows.")
            st.subheader("Preview")
            st.dataframe(df_out.head(50), use_container_width=True)

        # Download
        buf = io.BytesIO()
        if output_kind == "csv":
            df_out.to_csv(buf, index=False)
            st.download_button(
                "Download processed CSV",
                data=buf.getvalue(),
                file_name="processed_mash_resolution.csv",
                mime="text/csv"
            )
        else:
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df_out.to_excel(writer, index=False, sheet_name="output")
            st.download_button(
                "Download processed Excel",
                data=buf.getvalue(),
                file_name="processed_mash_resolution.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error("Something went wrong while processing your file.")
        st.exception(e)
