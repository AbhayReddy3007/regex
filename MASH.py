# app.py
import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MASH Resolution Extractor", layout="wide")
st.title("MASH + Resolution Sentence & Number Extractor")

st.caption("Upload an Excel (.xlsx) or CSV with an **abstracts** column.")

uploaded = st.file_uploader("Upload file", type=["xlsx", "csv"])

# ---------- helpers ----------
def split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    # Simple sentence splitter; handles cases with no punctuation
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()] or [text.strip()]

def extract_numbers(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    # numbers like 12, 12.5, 1,234.56 with optional %
    raw = re.findall(r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?', text)
    out = []
    for x in raw:
        if x.endswith('%'):
            out.append(x[:-1].replace(',', '') + '%')
        else:
            out.append(x.replace(',', ''))
    return out

# ---------- UI + processing ----------
if uploaded is None:
    st.info("⬆️ Choose a file to begin.")
else:
    try:
        # Read file
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.write("**Columns detected:**", list(df.columns))

        if 'abstracts' not in df.columns:
            st.error("The file must contain a column named **abstracts**.")
            st.stop()

        # Build 'sentences' column: sentences containing BOTH 'MASH' and 'resolution' (case-insensitive)
        sent_lists = df['abstracts'].apply(split_sentences)
        filtered = sent_lists.apply(
            lambda lst: [s for s in lst
                         if re.search(r'\bMASH\b', s, re.IGNORECASE)
                         and re.search(r'\bresolution\b', s, re.IGNORECASE)]
        )
        df['sentences'] = filtered.apply(lambda lst: " || ".join(lst))

        # Extract numbers from those sentences
        df['extracted values'] = df['sentences'].apply(
            lambda s: ", ".join(extract_numbers(s)) if isinstance(s, str) else ""
        )

        # Drop rows with empty 'sentences'
        has_sentences = df['sentences'].astype(str).str.strip().astype(bool)
        df_out = df[has_sentences].copy()

        if df_out.empty:
            st.warning("No sentences found containing both 'MASH' and 'resolution'.")
        else:
            st.success(f"Processed {len(df_out)} rows with matching sentences.")
            st.subheader("Preview")
            st.dataframe(df_out.head(50), use_container_width=True)

        # Download button
        buf = io.BytesIO()
        if uploaded.name.lower().endswith(".csv"):
            df_out.to_csv(buf, index=False)
            data = buf.getvalue()
            st.download_button(
                "Download processed CSV",
                data=data,
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
        # Show full traceback in the app so you never get a silent blank screen
        st.error("Something went wrong while processing your file.")
        st.exception(e)
