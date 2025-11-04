"""
streamlit_hba1c_extractor.py

Streamlit app that extracts **only the numbers present in the same sentence**
as an "HbA1c" or "A1c" mention. Uses regex only (no LLMs).

Save this file and run:
    streamlit run streamlit_hba1c_extractor.py
"""

import streamlit as st
import pandas as pd
import re
import math
from io import BytesIO

st.set_page_config(page_title="HbA1c Reduction Extractor", layout="wide")
st.title("HbA1c / A1c reduction extractor — numbers only from same sentence")

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
re_hba = re.compile(r'\b(?:hba1c|hb\s*a1c|a1c)\b', FLAGS)  # allow optional space in "hb a1c"

def parse_number(s: str) -> float:
    """Parse numbers that may use comma or dot as decimal separator."""
    if s is None:
        return float('nan')
    s = s.replace(',', '.').strip()
    try:
        return float(s)
    except:
        return float('nan')

def extract_hba1c_reductions(text: str):
    """
    Extract matches ONLY from sentences that contain 'hba1c' or 'a1c'.
    Sentences are split on ., ?, ! or newline. If a sentence contains 'hba1c'/'a1c',
    we search only that sentence for patterns and return matches. Otherwise return [].
    """
    if not isinstance(text, str):
        return []

    matches = []
    # conservative sentence split: split on punctuation or newline but keep content
    sentences = re.split(r'(?<=[\.\?!\n])\s+', text)

    for si, sent in enumerate(sentences):
        if not sent or not sent.strip():
            continue
        # Only process sentences mentioning HbA1c / A1c
        if not re_hba.search(sent):
            continue

        # 1) from X% to Y%  -> compute X - Y (absolute percentage points)
        for m in re_fromto.finditer(sent):
            a = parse_number(m.group(1))
            b = parse_number(m.group(2))
            reduction = None
            if not math.isnan(a) and not math.isnan(b):
                reduction = a - b
            matches.append({
                'raw': m.group(0),
                'type': 'from-to_same_sentence',
                'values': [a, b],
                'reduction_pp': reduction,
                'sentence_index': si,
                'span': m.span()
            })

        # 2) explicit "reduced by X%" or similar (treat as numeric reduction; ambiguous pp vs %)
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

        # 3) absolute phrasing like "reduction of 1.2 percentage points"
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

        # 4) ranges like "1.0-1.5%"
        for m in re_range.finditer(sent):
            a = parse_number(m.group(1))
            b = parse_number(m.group(2))
            # choose max value in the range as representative (you can change logic)
            rep = max(a, b) if not math.isnan(a) and not math.isnan(b) else None
            matches.append({
                'raw': m.group(0),
                'type': 'range_percent_same_sentence',
                'values': [a, b],
                'reduction_pp': rep,
                'sentence_index': si,
                'span': m.span()
            })

        # 5) any percent token in that sentence
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

    # If no matches found in any HbA1c-containing sentence, return empty list
    if not matches:
        return []

    # dedupe by (sentence_index, span)
    seen = set()
    filtered = []
    for mm in matches:
        key = (mm['sentence_index'], mm['span'])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(mm)

    # sort by sentence index then by match start
    filtered.sort(key=lambda x: (x['sentence_index'], x['span'][0] if isinstance(x['span'], tuple) else 0))
    return filtered

def choose_best_reduction(matches):
    """
    Choose a single 'best' reduction from matches (if any).
    Strategy: pick the largest absolute numeric reduction_pp.
    Returns (value_or_None, reason_type_or_empty).
    """
    best = None
    reason = ''
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
show_raw = st.sidebar.checkbox('Show raw matches column', value=True)
remove_nulls = st.sidebar.checkbox('Remove rows with no extracted reduction (best_reduction_pp is null)', value=True)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
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
    for idx, row in df.iterrows():
        text = row.get(text_col, '')
        matches = extract_hba1c_reductions(text)
        extracted_raws = [m['raw'] for m in matches]
        reductions = [m.get('reduction_pp') for m in matches]
        types = [m.get('type') for m in matches]
        best, reason = choose_best_reduction(matches)
        new = row.to_dict()
        new.update({
            'extracted_matches': extracted_raws,
            'reductions_pp': reductions,
            'reduction_types': types,
            'best_reduction_pp': best,
            'best_reduction_reason': reason
        })
        rows.append(new)
    return pd.DataFrame(rows)

out_df = process_df(df, col_name)

# Optionally remove rows with null best_reduction_pp (or empty extractions)
if remove_nulls:
    out_df = out_df[out_df['best_reduction_pp'].notna()].reset_index(drop=True)

# display
st.write('### Results (first 200 rows shown)')
display_df = out_df.copy()
if not show_raw:
    for c in ['extracted_matches', 'reductions_pp', 'reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

st.dataframe(display_df.head(200))

# allow download
@st.cache_data
def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(out_df)

st.download_button('Download results as Excel', data=excel_bytes, file_name='results_with_hba1c.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.markdown('---')
st.write('**Notes:**')
st.write('- This app returns ONLY numbers/percent tokens that are present in the same sentence as an HbA1c/A1c mention.')
st.write('- `reductions_pp` contains the numeric values extracted. For `from X% to Y%` patterns, the app computes the absolute difference (X - Y) as percentage points.')
st.write('- If no sentence in an abstract mentions HbA1c/A1c, that row will have no extracted matches (and will be removed if you checked the Remove rows option).')

st.write('')
st.write('Made for you — drop your file above and download the cleaned results.')
