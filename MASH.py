# app.py
# Streamlit app: Upload file -> create 'sentences' (MASH + resolution) -> 'extracted values' (numbers) -> drop empty rows -> download

import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MASH Resolution Extractor", layout="wide")
st.title("MASH + Resolution Sentence & Number Extractor")
st.caption("Upload an Excel (.xlsx) or CSV with an **abstracts** column.")

uploaded = st.file_uploader("Upload file", type=["xlsx", "csv"])

# ---------------- helpers ----------------
def split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    # Simple sentence splitter; includes last fragment if no terminal punctuation
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [text.strip()]

def extract_numbers(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    # numbers like 12, 12.5, 1,234.56 with optional trailing %
    raw = re.findall(r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?', text)
    out = []
    for x in raw:
        if x.endswith('%'):
            out.append(x[:-1].replace(',', '') + '%')
        else:
            out.append(x.replace(',', ''))
    return out

def build_sentences_column(series: pd.Series) -> pd.Series:
    sent_lists = series.apply(split_sentences)
    filtered = sent_lists.apply(
        lambda lst: [s for s in lst
                     if re.search(r'\bMASH\b', s, re.IGNORECASE)
                     and re.search(r'\bresolution\b', s, re.IGNORECASE)]
    )
    return filtered.apply(lambda lst: " || ".join(lst))

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

        # Ensure text type to avoid issues
        df['abstracts'] = df['abstracts'].astype(str)

        # 1) sentences column (must contain BOTH 'MASH' and 'resolution')
        df['sentences'] = build_sentences_column(df['abstracts'])

        # 2) extracted values column (all numeric tokens from sentences)
        df['extracted values'] = df['sentences'].apply(
            lambda s: ", ".join(extract_numbers(s)) if isinstance(s, str) else ""
        )

        # 3) drop rows where sentences is empty
        has_sentences = df['sentences'].astype(str).str.strip().astype(bool)
        df = df[has_sentences].copy()

        # 4) drop rows where extracted values is empty
        has_values = df['extracted values'].astype(str).str.strip().astype(bool)
        df_out = df[has_values].copy()

        if df_out.empty:
            st.warning("No rows left after filtering: either no matching sentences, or no numeric values in them.")
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
