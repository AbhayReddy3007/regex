# app.py
import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MASH Resolution Extractor", layout="wide")
st.title("MASH + Resolution Sentence & Number Extractor")

uploaded = st.file_uploader("Upload an Excel file (.xlsx) with an 'abstracts' column", type=["xlsx"])

def split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    # Basic sentence splitter; keeps URLs/decimals intact reasonably well
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # If the whole abstract has no sentence punctuation, treat it as one sentence
    if len(parts) == 1:
        return [text.strip()]
    return [p.strip() for p in parts if p.strip()]

def extract_numbers(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    # Match numbers like 12, 12.5, 1,234.56, and optional trailing %
    raw = re.findall(r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?', text)
    # Normalize commas in thousands groups (e.g., 1,234 -> 1234)
    cleaned = []
    for x in raw:
        if x.endswith('%'):
            num = x[:-1].replace(',', '')
            cleaned.append(num + '%')
        else:
            cleaned.append(x.replace(',', ''))
    return cleaned

if uploaded:
    df = pd.read_excel(uploaded)

    if 'abstracts' not in df.columns:
        st.error("The file must contain a column named 'abstracts'.")
        st.stop()

    # Build sentences column: sentences that contain BOTH 'MASH' and 'resolution' (case-insensitive)
    sent_lists = df['abstracts'].apply(split_sentences)
    mask_both = sent_lists.apply(
        lambda lst: [s for s in lst if re.search(r'\bMASH\b', s, re.IGNORECASE) and re.search(r'\bresolution\b', s, re.IGNORECASE)]
    )
    # Join with a clear separator so it's Excel-friendly
    df['sentences'] = mask_both.apply(lambda lst: " || ".join(lst))

    # Extract numbers from the 'sentences' text
    df['extracted values'] = df['sentences'].apply(lambda s: ", ".join(extract_numbers(s)))

    # Drop rows where 'sentences' is empty
    df_out = df[df['sentences'].str.strip().astype(bool)].copy()

    st.subheader("Preview")
    st.dataframe(df_out.head(50), use_container_width=True)

    # Download button
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, index=False, sheet_name="output")
    st.download_button("Download processed Excel", data=buf.getvalue(), file_name="processed_mash_resolution.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
