"""
streamlit_hba1c_extractor.py

Rows are kept ONLY if there's at least one sentence that:
  (1) mentions HbA1c/A1c, and
  (2) contains a number.
The 'sentence' column shows those sentence(s) (joined by " | " if multiple).
Numbers are extracted ONLY from those sentence(s). Regex only (no LLMs).
"""

import streamlit as st
import pandas as pd
import re
import math
from io import BytesIO

st.set_page_config(page_title="HbA1c Reduction Extractor", layout="wide")
st.title("HbA1c / A1c — show rows only when a sentence has HbA1c/A1c + a number")

# -------------------- regex helpers --------------------
NUM = r'(?:\d+(?:[.,]\d+)?)'
PCT = rf'({NUM})\s*%'
PP_WORD = r'(?:percentage points?|pp\b|points?)'
FROM_TO = rf'from\s+({NUM})\s*%\s*(?:to|->|−|—|-)\s*({NUM})\s*%'
REDUCE_BY = rf'(?:reduc(?:e|ed|tion|ed by|ing)|decrease(?:d)?|drop(?:ped)?|fell|reduced)\s*(?:by\s*)?({NUM})\s*(?:%|\s*{PP_WORD})'
ABS_PP = rf'(?:absolute\s+reduction\s+of|reduction\s+of)\s*({NUM})\s*(?:%|\s*{PP_WORD})'
RANGE_PCT = rf'({NUM})\s*(?:–|-|—)\s*({NUM})\s*%'

FLAGS = re.IGNORECASE
re_pct = re.compile(PCT, FLAGS)
re_fromto = re.compile(FROM_TO, FLAGS)
re_reduce_by = re.compile(REDUCE_BY, FLAGS)
re_abs_pp = re.compile(ABS_PP, FLAGS)
re_range = re.compile(RANGE_PCT, FLAGS)
re_hba = re.compile(r'\b(?:hba1c|hb\s*a1c|a1c)\b', FLAGS)
re_any_digit = re.compile(r'\d')

def parse_number(s: str) -> float:
    if s is None:
        return float('nan')
    s = s.replace(',', '.').strip()
    try:
        return float(s)
    except:
        return float('nan')

def split_sentences(text: str):
    """Conservative sentence splitter on ., ?, ! or newlines."""
    text = str(text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(?<=[\.\?!])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def extract_hba1c_same_sentence(text: str):
    """
    Return (matches, sentences_used)
    - matches: list of dicts for numbers captured ONLY from sentences that
      contain HbA1c/A1c AND at least one digit.
    - sentences_used: list of those sentence strings.
    """
    if not isinstance(text, str):
        return [], []

    matches, sentences_used = [], []
    sentences = split_sentences(text)

    for si, sent in enumerate(sentences):
        if not re_hba.search(sent):
            continue
        if not re_any_digit.search(sent):
            continue  # must have a number in that sentence

        sentences_used.append(sent)

        # from X% to Y%  -> compute X - Y (percentage points)
        for m in re_fromto.finditer(sent):
            a = parse_number(m.group(1))
            b = parse_number(m.group(2))
            reduction = None if math.isnan(a) or math.isnan(b) else (a - b)
            matches.append({
                'raw': m.group(0),
                'type': 'from-to_same_sentence',
                'values': [a, b],
                'reduction_pp': reduction,
                'sentence_index': si,
                'span': m.span()
            })

        # explicit "reduced by X%" or similar
        for m in re_reduce_by.finditer(sent):
            v = parse_number(m.group(1))
            matches.append({
                'raw': m.group(0),
                'type': 'percent_or_pp_same_sentence',
                'values': [v],
                'reduction_pp': v,
                'sentence_index': si,
                'span': m.span()
            })

        # absolute phrasing like "reduction of 1.2 percentage points"
        for m in re_abs_pp.finditer(sent):
            v = parse_number(m.group(1))
            matches.append({
                'raw': m.group(0),
                'type': 'pp_word_same_sentence',
                'values': [v],
                'reduction_pp': v,
                'sentence_index': si,
                'span': m.span()
            })

        # ranges like "1.0–1.5%"
        for m in re_range.finditer(sent):
            a = parse_number(m.group(1))
            b = parse_number(m.group(2))
            rep = None if math.isnan(a) or math.isnan(b) else max(a, b)
            matches.append({
                'raw': m.group(0),
                'type': 'range_percent_same_sentence',
                'values': [a, b],
                'reduction_pp': rep,
                'sentence_index': si,
                'span': m.span()
            })

        # any percent token in that sentence
        for m in re_pct.finditer(sent):
            v = parse_number(m.group(1))
            matches.append({
                'raw': m.group(0),
                'type': 'percent_same_sentence',
                'values': [v],
                'reduction_pp': v,
                'sentence_index': si,
                'span': m.span()
            })

    # dedupe by (sentence_index, span)
    seen, filtered = set(), []
    for mm in matches:
        key = (mm['sentence_index'], mm['span'])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(mm)

    filtered.sort(key=lambda x: (x['sentence_index'], x['span'][0]))
    return filtered, sentences_used

def choose_best_reduction(matches):
    """Pick the largest absolute reduction_pp from matches, if any."""
    best, reason = None, ''
    for m in matches:
        v = m.get('reduction_pp')
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if best is None or abs(v) > abs(best):
            best = v
            reason = m.get('type', '')
    return best, reason

# -------------------- UI --------------------

st.sidebar.header('Options')
uploaded = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name = st.sidebar.text_input('Column that contains abstracts/text', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (leave blank for first sheet)', value='')
show_raw = st.sidebar.checkbox('Show raw matches columns', value=False)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".")
    st.stop()

# read file
try:
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        if sheet_name:
            df = pd.read_excel(uploaded, sheet_name=sheet_name)
        else:
            df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

st.success(f'Loaded {len(df)} rows. Processing...')

@st.cache_data
def process_df(df, text_col):
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        matches, sentences_used = extract_hba1c_same_sentence(text)

        new = row.to_dict()
        new.update({
            'sentence': ' | '.join(sentences_used) if sentences_used else '',
            'extracted_matches': [m['raw'] for m in matches],
            'reductions_pp': [m.get('reduction_pp') for m in matches],
            'reduction_types': [m.get('type') for m in matches],
        })
        best, reason = choose_best_reduction(matches)
        new.update({
            'best_reduction_pp': best,
            'best_reduction_reason': reason
        })
        rows.append(new)

    out = pd.DataFrame(rows)
    # Keep ONLY rows where at least one qualifying sentence exists
    out = out[out['sentence'].astype(str).str.len() > 0].reset_index(drop=True)
    return out

out_df = process_df(df, col_name)

# display
st.write('### Results (first 200 rows shown)')
display_df = out_df.copy()
if not show_raw:
    for c in ['extracted_matches', 'reductions_pp', 'reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

st.dataframe(display_df.head(200))

# download
@st.cache_data
def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(out_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_hba1c.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

st.markdown('---')
st.write('**Rules applied:**')
st.write('- A row appears **only** if there is a sentence that contains HbA1c/A1c **and** a number.')
st.write('- The **sentence** column shows exactly that/those sentence(s) (joined with ` | ` if multiple).')
st.write('- Extracted numbers are restricted to those sentence(s); `best_reduction_pp` is derived from them.')
