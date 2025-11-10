"""
HbA1c / A1c % reduction extractor (regex-only)

Rules:
- Keep a row only if at least one SENTENCE:
  (1) mentions HbA1c OR A1c,
  (2) contains a % number,
  (3) includes a reduction cue (reduced/decreased/lowered/dropped/fell/declined or 'from ... to ...').

- Within each qualifying sentence, take ONLY the % values within the NEXT 4 WORDS
  after each HbA1c/A1c mention (avoids unrelated values like body-weight %).

- Strict numeric filter: keep ONLY values < 7.
  * For from→to, the extracted value is (X − Y) and must also be < 7.

Outputs per row:
- sentence (the qualifying sentence(s))
- extracted_matches (delta% for from→to, raw % for others)
- reductions_pp (numbers used, including computed delta)
- reduction_types
"""

import streamlit as st
import pandas as pd
import re
import math
from io import BytesIO

st.set_page_config(page_title="HbA1c/A1c % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c — % reductions within next 4 words (strict < 7)")

# -------------------- Regex helpers --------------------
NUM = r'(?:\d+(?:[.,]\d+)?)'
PCT = rf'({NUM})\s*%'
DASH = r'(?:-|–|—)'  # '-', en dash, em dash

# Reduction patterns (all require %)
FROM_TO   = rf'from\s+({NUM})\s*%\s*(?:to|->|{DASH})\s*({NUM})\s*%'
REDUCE_BY = rf'(?:reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\s*(?:by\s*)?({NUM})\s*%'
ABS_PP    = rf'(?:absolute\s+reduction\s+of|reduction\s+of)\s*({NUM})\s*%'
RANGE_PCT = rf'({NUM})\s*{DASH}\s*({NUM})\s*%'

FLAGS = re.IGNORECASE
re_pct       = re.compile(PCT, FLAGS)
re_fromto    = re.compile(FROM_TO, FLAGS)
re_reduce_by = re.compile(REDUCE_BY, FLAGS)
re_abs_pp    = re.compile(ABS_PP, FLAGS)
re_range     = re.compile(RANGE_PCT, FLAGS)

# HbA1c terms & cues
re_hba1c = re.compile(r'\bhb\s*a1c\b|\bhba1c\b|\ba1c\b', FLAGS)
re_reduction_cue = re.compile(
    r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b',
    FLAGS
)

# -------------------- Utilities --------------------
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
    has_term = bool(re_hba1c.search(sent))
    has_pct  = bool(re_pct.search(sent))
    has_cue  = bool(re_reduction_cue.search(sent))
    return has_term and has_pct and has_cue

# -------- HbA1c-tied extraction: NEXT 4 WORDS after HbA1c mention --------
def _next_n_words_window(s: str, start: int, n: int = 4):
    """Return (segment, abs_start, abs_end) covering the next n tokens (\S+) after index 'start'."""
    sub = s[start:]
    end_rel = 0
    count = 0
    for m in re.finditer(r'\S+', sub):
        end_rel = m.end()
        count += 1
        if count >= n:
            break
    if count == 0:
        return '', start, start
    abs_start = start
    abs_end = start + end_rel
    return s[abs_start:abs_end], abs_start, abs_end

def _add(match_list, si, seg_abs_start, m, typ, values, reduction):
    match_list.append({
        'raw': m.group(0),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (seg_abs_start + m.start(), seg_abs_start + m.end())
    })

def extract_next4_in_sentence(sent: str, si: int):
    """Within a sentence, take only matches inside the next 4-word window after each HbA1c/A1c hit."""
    matches = []
    for hh in re_hba1c.finditer(sent):
        seg, abs_s, _ = _next_n_words_window(sent, hh.end(), 4)
        if not seg:
            continue
        # from X% to Y%
        for m in re_fromto.finditer(seg):
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            red = None if math.isnan(a) or math.isnan(b) else (a - b)
            _add(matches, si, abs_s, m, 'from-to_next4', [a, b], red)
        # reduced/decreased/... by X%
        for m in re_reduce_by.finditer(seg):
            v = parse_number(m.group(1))
            _add(matches, si, abs_s, m, 'percent_or_pp_next4', [v], v)
        # reduction of X%
        for m in re_abs_pp.finditer(seg):
            v = parse_number(m.group(1))
            _add(matches, si, abs_s, m, 'pp_word_next4', [v], v)
        # ranges 1.0–1.5%
        for m in re_range.finditer(seg):
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            rep = None if math.isnan(a) or math.isnan(b) else max(a, b)
            _add(matches, si, abs_s, m, 'range_percent_next4', [a, b], rep)
        # any % token
        for m in re_pct.finditer(seg):
            v = parse_number(m.group(1))
            _add(matches, si, abs_s, m, 'percent_next4', [v], v)

    # dedupe by span inside sentence
    seen = set(); out = []
    for mm in matches:
        if mm['span'] in seen:
            continue
        seen.add(mm['span'])
        out.append(mm)
    out.sort(key=lambda x: x['span'][0])
    return out

def extract_hba1c_sentences_next4(text: str):
    """Return (matches, sentences_used) for sentences meeting the criterion, using next-4-words windows."""
    matches, sentences_used = [], []
    for si, sent in enumerate(split_sentences(text)):
        if not sentence_meets_criterion(sent):
            continue
        sentences_used.append(sent)
        matches.extend(extract_next4_in_sentence(sent, si))

    # dedupe globally by (sentence_index, span)
    seen, filtered = set(), []
    for mm in matches:
        key = (mm['sentence_index'], mm['span'])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(mm)

    filtered.sort(key=lambda x: (x['sentence_index'], x['span'][0]))
    return filtered, sentences_used

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded     = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name     = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name   = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug   = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named \"abstract\".')
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

        matches, sentences_used = extract_hba1c_sentences_next4(text)

        # STRICT FILTER: keep only values < 7 (prefer computed reduction_pp if present)
        def _allowed(m):
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                return float(rp) < 7.0
            vals = m.get('values') or []
            for v in vals:
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) < 7.0:
                        return True
                except Exception:
                    pass
            return False

        matches = [m for m in matches if _allowed(m)]

        # formatting: show delta% for from→to, raw % otherwise
        def _fmt_pct(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ''
            s = f\"{float(v):.3f}\".rstrip('0').rstrip('.')
            return f\"{s}%\"
        def _fmt_extracted(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return _fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        new = row.to_dict()
        new.update({
            'sentence': ' | '.join(sentences_used) if sentences_used else '',
            'extracted_matches': [_fmt_extracted(m) for m in matches],
            'reductions_pp': [m.get('reduction_pp') for m in matches],
            'reduction_types': [m.get('type') for m in matches],
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # Keep ONLY rows where at least one qualifying sentence exists
    out = out[out['sentence'].astype(str).str.len() > 0]

    # Drop rows where extracted_matches is empty or not a list
    out = out[out['extracted_matches'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    return out.reset_index(drop=True)

out_df = process_df(df, col_name)

# display
st.write('### Results (first 200 rows shown)')
display_df = out_df.copy()
# Always show 'extracted_matches' and 'sentence'; optionally hide debug columns
if not show_debug:
    for c in ['reductions_pp', 'reduction_types']:
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
st.write('**Rules enforced:**')
st.write('- Sentence must contain HbA1c **or** A1c, a **% number**, and a **reduction cue**.')
st.write('- Only % values within the **next 4 words** after HbA1c/A1c are considered.')
st.write('- **Strict filter**: keep only values **< 7**. For from→to, the extracted value is the **delta** and must also be < 7.')
