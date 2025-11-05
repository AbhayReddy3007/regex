"""
streamlit_hba1c_extractor.py

Keeps a row ONLY if at least one sentence:
  (1) mentions HbA1c OR A1c,
  (2) contains a percentage value (e.g., 7.8%),
  (3) and includes a reduction cue (e.g., reduced/reduction/decreased/lowered/dropped/fell or 'from ... to ...').

In extracted_matches we show only:
  - % values < 8, OR
  - any explicit "from X% to Y%" pattern (kept even if X or Y ≥ 8).

The 'sentence' column shows those sentence(s), joined by " | " if multiple.
Regex only (no LLMs).
"""

import streamlit as st
import pandas as pd
import re
import math
from io import BytesIO

st.set_page_config(page_title="HbA1c/A1c % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c — % reductions with cue words (keeps `sentence` column)")

# -------------------- Regex helpers --------------------
NUM = r'(?:\d+(?:[.,]\d+)?)'
PCT = rf'({NUM})\s*%'
# Use actual en/em dashes to avoid encoding issues
DASH = r'(?:-|–|—)'

# Reduction patterns (all require %)
FROM_TO   = rf'from\s+({NUM})\s*%\s*(?:to|->|{DASH})\s*({NUM})\s*%'
REDUCE_BY = rf'(?:reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?)\s*(?:by\s*)?({NUM})\s*%'
ABS_PP    = rf'(?:absolute\s+reduction\s+of|reduction\s+of)\s*({NUM})\s*%'
RANGE_PCT = rf'({NUM})\s*{DASH}\s*({NUM})\s*%'

FLAGS = re.IGNORECASE
re_pct       = re.compile(PCT, FLAGS)
re_fromto    = re.compile(FROM_TO, FLAGS)
re_reduce_by = re.compile(REDUCE_BY, FLAGS)
re_abs_pp    = re.compile(ABS_PP, FLAGS)
re_range     = re.compile(RANGE_PCT, FLAGS)

# Terms
re_hba1c_word = re.compile(r'\bhb\s*a1c\b|\bhba1c\b', FLAGS)  # allows "HB A1c" and "HbA1c"
re_a1c_word   = re.compile(r'\ba1c\b', FLAGS)

# Reduction cue (includes 'from' so we catch "from X% to Y%")
re_reduction_cue = re.compile(
    r'\b(reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing)|from)\b',
    FLAGS
)

def parse_number(s: str) -> float:
    if s is None:
        return float('nan')
    s = s.replace(',', '.').strip()
    try:
        return float(s)
    except Exception:
        return float('nan')

def split_sentences(text: str):
    """Conservative sentence splitter on ., ?, ! or newlines."""
    text = '' if not isinstance(text, str) else text
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(?<=[\.!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def sentence_meets_criterion(sent: str) -> bool:
    """Require: (HbA1c OR A1c) AND a % AND a reduction cue."""
    has_term = bool(re_hba1c_word.search(sent) or re_a1c_word.search(sent))
    has_pct  = bool(re_pct.search(sent))
    has_cue  = bool(re_reduction_cue.search(sent))
    return has_term and has_pct and has_cue

def extract_from_sentence(sent: str, si: int):
    """Extract reduction-related % patterns only from the given sentence."""
    matches = []

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

    # explicit "... by X%"
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

    # "... reduction of X%"
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

    # any % token in that cue sentence (e.g., "HbA1c reduced to 7.1%")
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

    return matches

def extract_hba1c_sentences_with_percent_reduction(text: str):
    """
    Return (matches, sentences_used) for sentences that pass the stricter criterion.
    """
    matches, sentences_used = [], []
    for si, sent in enumerate(split_sentences(text)):
        if not sentence_meets_criterion(sent):
            continue
        sentences_used.append(sent)
        matches.extend(extract_from_sentence(sent, si))

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
uploaded     = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name     = st.sidebar.text_input('Column that contains abstracts/text', value='abstract')
sheet_name   = st.sidebar.text_input('Excel sheet name (leave blank for first sheet)', value='')
show_raw     = st.sidebar.checkbox('Show raw matches columns', value=False)
require_best = st.sidebar.checkbox('Also require a computed best_reduction_pp', value=False)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named \"abstract\".")
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
    st.error(f'Column \"{col_name}\" not found. Available columns: {list(df.columns)}')
    st.stop()

st.success(f'Loaded {len(df)} rows. Processing...')

@st.cache_data
def process_df(df, text_col):
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        text = '' if not isinstance(text, str) else text

        matches, sentences_used = extract_hba1c_sentences_with_percent_reduction(text)

        # -- Filter for extracted_matches: keep values < 8, OR any explicit from-to --
        def _allowed(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return True
            vals = m.get('values') or []
            for v in vals:
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) < 8.0:
                        return True
                except Exception:
                    pass
            rp = m.get('reduction_pp')
            try:
                if rp is not None and not (isinstance(rp, float) and math.isnan(rp)) and float(rp) < 8.0:
                    return True
            except Exception:
                pass
            return False

        matches = [m for m in matches if _allowed(m)]

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
    out = out[out['sentence'].astype(str).str.len() > 0]

    # Drop rows where extracted_matches is empty or not a list
    out = out[out['extracted_matches'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    # Optionally also require best_reduction_pp
    if require_best:
        out = out[out['best_reduction_pp'].notna()]

    return out.reset_index(drop=True)

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
st.write('- Sentence must contain HbA1c **or** A1c, a **% number**, and a **reduction cue** (reduced / from ... to ... / decreased / lowered / dropped / fell / declined).')
st.write('- The **sentence** column shows exactly those sentences; rows without such sentences are dropped.')
st.write('- In `extracted_matches`, we keep only % values < 8 or any explicit from-to pattern.')
