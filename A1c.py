# streamlit_a1c_duration_extractor.py
"""
Streamlit app: HbA1c (A1c) extraction + scoring + duration column.
Saves only the A1c-related extraction and score logic (no weight, no LLM).
Drop-in replacement when you only need A1c + duration.

Usage: streamlit run streamlit_a1c_duration_extractor.py
"""

import re
import math
from io import BytesIO
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HbA1c + Duration Extractor", layout="wide")
st.title("HbA1c (A1c) — regex extraction + score + duration")

# -------------------- Regex helpers --------------------
NUM = r'(?:[+-]?\d+(?:[.,·]\d+)?)'
PCT = rf'({NUM})\s*%'
DASH = r'(?:-|–|—)'

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

re_hba1c  = re.compile(r'\bhb\s*a1c\b|\bhba1c\b|\ba1c\b', FLAGS)
re_reduction_cue = re.compile(r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b', FLAGS)

# Duration regex (same normalisation used previously)
DURATION_RE = re.compile(
    r'\b(?:T\d{1,2}|'                                       # T6, T12
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:weeks?|wks?|wk|w)\b|'   # 12 weeks, 6-12 weeks
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:months?|mos?|mo)\b|'    # 6 months, 12-mo, 6-12 months
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:days?|d)\b|'            # days
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:years?|yrs?)\b|'        # years
    r'\d{1,3}-week\b|\d{1,3}-month\b|\d{1,3}-mo\b)',         # hyphenated forms
    FLAGS
)

# -------------------- Utilities --------------------

def parse_number(s: str) -> float:
    if s is None:
        return float('nan')
    s = str(s).replace(',', '.').replace('·', '.').strip()
    try:
        return float(s)
    except Exception:
        return float('nan')


def split_sentences(text: str):
    if not isinstance(text, str):
        return []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]


def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.2f}".rstrip('0').rstrip('.')
    return f"{s}%"


def extract_durations(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    found = []
    seen = set()
    for m in DURATION_RE.finditer(text):
        token = m.group(0).strip()
        token = re.sub(r'\s+', ' ', token)
        token = token.replace('–', '-').replace('—', '-')
        token = re.sub(r'\bmos?\b', 'months', token, flags=re.IGNORECASE)
        token = re.sub(r'\bmo\b', 'months', token, flags=re.IGNORECASE)
        token = re.sub(r'\bwks?\b', 'weeks', token, flags=re.IGNORECASE)
        token = re.sub(r'\bw\b', 'weeks', token, flags=re.IGNORECASE)
        token = re.sub(r'\bd\b', 'days', token, flags=re.IGNORECASE)
        token = re.sub(r'\byrs?\b', 'years', token, flags=re.IGNORECASE)
        token = token.strip()
        if token.lower() not in seen:
            seen.add(token.lower())
            found.append(token)
    return ' | '.join(found)

# --- small window builder used to search near the target term (±5 spaces inclusive tokens) ---
def window_prev_next_spaces_inclusive_tokens(s: str, pos: int, n_prev_spaces: int = 5, n_next_spaces: int = 5):
    space_like = set([' ', '\t', '\n', '\r'])
    L = len(s)

    i = pos - 1
    spaces = 0
    left_boundary_start = pos
    while i >= 0 and spaces < n_prev_spaces:
        if s[i] in space_like:
            while i >= 0 and s[i] in space_like:
                i -= 1
            spaces += 1
            left_boundary_start = i + 1
        else:
            i -= 1
    j = left_boundary_start - 1
    while j >= 0 and s[j] not in space_like:
        j -= 1
    start = j + 1

    k = pos
    spaces = 0
    right_boundary_end = pos
    while k < L and spaces < n_next_spaces:
        if s[k] in space_like:
            while k < L and s[k] in space_like:
                k += 1
            spaces += 1
            right_boundary_end = k
        else:
            k += 1
    m = right_boundary_end
    while m < L and s[m] not in space_like:
        m += 1
    end = m

    start = max(0, min(start, L))
    end = max(start, min(end if end != pos else L, L))
    return s[start:end], start, end


def add_match(out, si, abs_start, m, typ, values, reduction):
    out.append({
        'raw': m.group(0) if hasattr(m, 'group') else str(m),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (abs_start + (m.start() if hasattr(m, 'start') else 0), abs_start + (m.end() if hasattr(m, 'end') else 0)),
    })

# -------------------- Core A1c extraction --------------------

def extract_in_sentence(sent: str, si: int):
    matches = []
    # 1) from-to whole-sentence
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red_pp = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        rel = None
        if not (math.isnan(a) or math.isnan(b)) and b != 0:
            rel = ((a - b) / b) * 100.0
        add_match(matches, si, 0, m, 'from-to_sentence', [a, b], red_pp)
        if rel is not None:
            rel_raw = f"{rel:.6f}%"
            matches.append({
                'raw': rel_raw,
                'type': 'from-to_relative_percent',
                'values': [a, b, rel],
                'reduction_pp': red_pp,
                'sentence_index': si,
                'span': (m.start(), m.end()),
            })

    # 2) ±5-spaces window around each A1c occurrence
    any_window_hit = False
    for hh in re_hba1c.finditer(sent):
        seg, abs_s, _ = window_prev_next_spaces_inclusive_tokens(sent, hh.end(), 5, 5)

        for m in re_reduce_by.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, 'percent_or_pp_pmSpaces5', [v], v)

        for m in re_abs_pp.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, 'pp_word_pmSpaces5', [v], v)

        for m in re_range.finditer(seg):
            any_window_hit = True
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            rep = None if (math.isnan(a) or math.isnan(b)) else max(a, b)
            add_match(matches, si, abs_s, m, 'range_percent_pmSpaces5', [a, b], rep)

        for m in re_pct.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, 'percent_pmSpaces5', [v], v)

    # dedupe by span
    seen = set()
    uniq = []
    for mm in matches:
        if mm['span'] in seen:
            continue
        seen.add(mm['span'])
        uniq.append(mm)
    uniq.sort(key=lambda x: x['span'][0])
    return uniq


def sentence_meets_criterion(sent: str) -> bool:
    has_term = bool(re_hba1c.search(sent))
    has_pct = bool(re_pct.search(sent))
    has_cue = bool(re_reduction_cue.search(sent))
    return has_term and has_pct and has_cue


def extract_sentences(text: str):
    matches, sentences_used = [], []
    for si, sent in enumerate(split_sentences(text)):
        if not sentence_meets_criterion(sent):
            continue
        sentences_used.append(sent)
        matches.extend(extract_in_sentence(sent, si))

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

# -------------------- Scoring --------------------

def compute_a1c_score(selected_pct_str: str):
    """Scores for A1c:
    5: >2.2%
    4: 1.8%-2.1%
    3: 1.2%-1.7%
    2: 0.8%-1.1%
    1: <0.8%
    """
    if not selected_pct_str:
        return ""
    try:
        val = parse_number(selected_pct_str.replace('%', ''))
    except:
        return ""
    if val > 2.2:
        return 5
    if 1.8 <= val <= 2.1:
        return 4
    if 1.2 <= val <= 1.7:
        return 3
    if 0.8 <= val <= 1.1:
        return 2
    if val < 0.8:
        return 1
    return ""

# -------------------- Processing --------------------
@st.cache_data
def process_df(df_in: pd.DataFrame, text_col: str):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, '')
        if not isinstance(text_orig, str):
            text_orig = '' if pd.isna(text_orig) else str(text_orig)

        duration_str = extract_durations(text_orig)
        hba_matches, hba_sentences = extract_sentences(text_orig)

        # precompute relative from-to percent if present
        def _find_relative_fromto(matches):
            for m in matches:
                t = (m.get('type') or '').lower()
                if 'from-to_relative_percent' in t or 'from-to_relative' in t:
                    vals = m.get('values') or []
                    if len(vals) >= 3 and vals[2] is not None and not math.isnan(vals[2]):
                        return fmt_pct(vals[2])
            for m in matches:
                t = (m.get('type') or '').lower()
                if 'from-to_sentence' in t:
                    vals = m.get('values') or []
                    if len(vals) >= 2:
                        a = vals[0]; b = vals[1]
                        if not (math.isnan(a) or math.isnan(b)) and b != 0:
                            rel = ((a - b) / b) * 100.0
                            return fmt_pct(rel)
            return None

        precomputed_hba_rel = _find_relative_fromto(hba_matches)

        def fmt_extracted(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                if 'relative' in t:
                    vals = m.get('values') or []
                    if len(vals) >= 3 and vals[2] is not None and not math.isnan(vals[2]):
                        return fmt_pct(vals[2])
                    if m.get('reduction_pp') is not None and not math.isnan(m.get('reduction_pp')):
                        return fmt_pct(m.get('reduction_pp'))
                    return m.get('raw', '')
                else:
                    if m.get('reduction_pp') is not None and not math.isnan(m.get('reduction_pp')):
                        return fmt_pct(m.get('reduction_pp'))
                    return m.get('raw', '')
            if isinstance(m.get('reduction_pp'), (int, float)) and not math.isnan(m.get('reduction_pp')):
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        hba_regex_vals = [fmt_extracted(m) for m in hba_matches]

        # Choose selected percent: prefer precomputed relative if available, else first percent candidate
        selected = ''
        if precomputed_hba_rel:
            selected = precomputed_hba_rel
        else:
            # pick the first percent-looking item in regex_vals
            for it in hba_regex_vals:
                if isinstance(it, str) and it.strip().endswith('%'):
                    # normalise
                    s2 = it.replace('%', '').replace(',', '.').strip()
                    try:
                        v = float(s2)
                        selected = fmt_pct(abs(v))
                        break
                    except:
                        continue

        # compute score
        a1c_score = compute_a1c_score(selected)

        new = row.to_dict()
        new.update({
            'sentence': ' | '.join(hba_sentences) if hba_sentences else '',
            'extracted_matches': hba_regex_vals,
            'selected %': selected,
            'A1c Score': a1c_score,
            'duration': duration_str,
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # keep rows that have extraction
    def has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(has_items))
    out = out[mask_hba].reset_index(drop=True)

    out.attrs['counts'] = dict(kept=int(mask_hba.sum()), total=int(len(out)))
    return out

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
show_debug = st.sidebar.checkbox('Show debug columns (extracted_matches, sentence)', value=True)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
    st.stop()

try:
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, sheet_name=0)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

st.success(f'Loaded {len(df)} rows. Processing...')

out_df = process_df(df, col_name)

# reorder so duration is last
if 'duration' in out_df.columns:
    cols_no_duration = [c for c in out_df.columns if c != 'duration']
    cols_no_duration.append('duration')
    out_df = out_df[cols_no_duration]

st.write("### Results (first 200 rows shown)")
if not show_debug:
    for c in ['extracted_matches', 'sentence']:
        if c in out_df.columns:
            out_df = out_df.drop(columns=[c])

st.dataframe(out_df.head(200))

counts = out_df.attrs.get('counts', None)
if counts:
    kept = counts.get('kept', 0)
    st.caption(f"Kept {kept} rows with A1c extraction.")

# Download
@st.cache_data
def to_excel_bytes(df_out):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(out_df)
st.download_button('Download results as Excel', data=excel_bytes, file_name='a1c_results_with_duration.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
