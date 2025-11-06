"""
Weight Loss % extractor (regex-only)

Rules:
- Keep a row only if at least one SENTENCE:
  (1) mentions weight (weight/body weight/weight loss),
  (2) contains a % number,
  (3) includes a reduction cue (lost/reduced/decreased/lowered/dropped/fell/declined or 'from ... to ...').

- Within each qualifying sentence, keep only % values NEAR the weight mention (proximity window)
  so unrelated values like HbA1c (-0.9%) are ignored.

- For from→to, the extracted value is the difference (X − Y) in percentage points.

Outputs per row:
- sentence (the qualifying sentence(s))
- extracted_matches (delta for from→to, raw % for others)
- reductions_pp (numbers)
- reduction_types
- best_reduction_pp
"""

import streamlit as st
import pandas as pd
import re
import math
from io import BytesIO

st.set_page_config(page_title="Weight Loss % Extractor", layout="wide")
st.title("Weight Loss — % reductions tied to weight terms")

# -------------------- Regex helpers --------------------
NUM  = r'(?:\d+(?:[.,]\d+)?)'
PCT  = rf'({NUM})\s*%'
DASH = r'(?:-|–|—)'  # '-', en dash, em dash

# Reduction patterns (all require %)
FROM_TO   = rf'from\s+({NUM})\s*%\s*(?:to|->|{DASH})\s*({NUM})\s*%'
REDUCE_BY = rf'(?:reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing)|lost)\s*(?:by\s*)?({NUM})\s*%'
ABS_PP    = rf'(?:absolute\s+reduction\s+of|reduction\s+of|loss\s+of)\s*({NUM})\s*%'
RANGE_PCT = rf'({NUM})\s*{DASH}\s*({NUM})\s*%'

FLAGS = re.IGNORECASE
re_pct       = re.compile(PCT, FLAGS)
re_fromto    = re.compile(FROM_TO, FLAGS)
re_reduce_by = re.compile(REDUCE_BY, FLAGS)
re_abs_pp    = re.compile(ABS_PP, FLAGS)
re_range     = re.compile(RANGE_PCT, FLAGS)

# Weight terms & cues
# Keep "weight" broad but rely on proximity windows to avoid unrelated percents
re_weight_terms = re.compile(r'\b(weight\s*loss|body\s*weight|weight|bw)\b', FLAGS)
re_reduction_cue_weight = re.compile(
    r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing)|lost|loss\s+of)\b',
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

def sentence_meets_criterion_weight(sent: str) -> bool:
    """Require: (weight term) AND a % AND a reduction cue."""
    has_term = bool(re_weight_terms.search(sent))
    has_pct  = bool(re_pct.search(sent))
    has_cue  = bool(re_reduction_cue_weight.search(sent))
    return has_term and has_pct and has_cue

# -------- Weight-tied extraction inside one sentence (proximity windows) --------
WT_SCOPE_LEFT  = 50   # chars to the left of weight hit
WT_SCOPE_RIGHT = 80   # chars to the right of weight hit

def _add(match_list, si, span_offset, m, typ, values, reduction):
    match_list.append({
        'raw': m.group(0),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (span_offset + m.start(), span_offset + m.end())
    })

def extract_weight_tied_matches_in_sentence(sent: str, si: int):
    """
    Extract ONLY % values that are tied to weight by proximity:
    For each weight mention, take a small window around it and search there.
    """
    matches = []
    hits = list(re_weight_terms.finditer(sent))
    if not hits:
        return matches

    for h in hits:
        s = max(0, h.start() - WT_SCOPE_LEFT)
        e = min(len(sent), h.end() + WT_SCOPE_RIGHT)
        seg = sent[s:e]

        # from X% to Y%  -> compute X - Y (percentage points)
        for m in re_fromto.finditer(seg):
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            red = None if math.isnan(a) or math.isnan(b) else (a - b)
            _add(matches, si, s, m, 'from-to_weight_scope', [a, b], red)

        # explicit "... by X%" or "lost X%"
        for m in re_reduce_by.finditer(seg):
            v = parse_number(m.group(1))
            _add(matches, si, s, m, 'percent_or_pp_weight_scope', [v], v)

        # "loss of X%" / "reduction of X%"
        for m in re_abs_pp.finditer(seg):
            v = parse_number(m.group(1))
            _add(matches, si, s, m, 'pp_word_weight_scope', [v], v)

        # ranges like "5–7%"
        for m in re_range.finditer(seg):
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            rep = None if math.isnan(a) or math.isnan(b) else max(a, b)
            _add(matches, si, s, m, 'range_percent_weight_scope', [a, b], rep)

        # any % token in the proximity window
        for m in re_pct.finditer(seg):
            v = parse_number(m.group(1))
            _add(matches, si, s, m, 'percent_weight_scope', [v], v)

    # dedupe by absolute span within the sentence
    seen = set()
    out = []
    for mm in matches:
        key = (mm['span'][0], mm['span'][1])
        if key in seen:
            continue
        seen.add(key)
        out.append(mm)

    out.sort(key=lambda x: x['span'][0])
    return out

def extract_weight_sentences(text: str):
    """
    Return (matches, sentences_used) for sentences that pass the weight criterion.
    Within those, only keep % values tied to weight via proximity windows.
    """
    matches, sentences_used = [], []
    sentences = split_sentences(text)
    for si, sent in enumerate(sentences):
        if not sentence_meets_criterion_weight(sent):
            continue
        sentences_used.append(sent)
        matches.extend(extract_weight_tied_matches_in_sentence(sent, si))

    # dedupe globally by sentence-span
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
col_name     = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name   = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')

# Optional cap (no cap by default)
cap_txt = st.sidebar.text_input('Optional MAX % cap (leave blank for no cap)', value='')
require_best = st.sidebar.checkbox('Also require best_reduction_pp to exist', value=False)
show_debug   = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

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

# -------------------- Processor --------------------
@st.cache_data
def process_df_weight(df, text_col, cap_val=None):
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        text = '' if not isinstance(text, str) else text

        matches, sentences_used = extract_weight_sentences(text)

        # Optional cap filter
        def _allowed(m):
            rp = m.get('reduction_pp')
            # prefer computed reduction if present (covers from→to and "... by X%"/"lost X%")
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                val = float(rp)
            else:
                # fallback to any % value captured
                val = None
                for v in (m.get('values') or []):
                    try:
                        if v is not None and not (isinstance(v, float) and math.isnan(v)):
                            val = float(v)
                            break
                    except Exception:
                        pass
                if val is None:
                    return False
            if cap_val is None:
                return True
            return val <= cap_val

        matches = [m for m in matches if _allowed(m)]

        # Display formatting: in from→to cases, show the delta only
        def _fmt_pct(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ''
            s = f"{float(v):.3f}".rstrip('0').rstrip('.')
            return f"{s}%"

        def _fmt_extracted(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return _fmt_pct(m.get('reduction_pp'))  # show the delta only
            return m.get('raw', '')

        new = row.to_dict()
        new.update({
            'sentence': ' | '.join(sentences_used) if sentences_used else '',
            'extracted_matches': [_fmt_extracted(m) for m in matches],
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

# Parse optional cap
cap_val = None
try:
    cap_val = float(cap_txt) if cap_txt.strip() != '' else None
except Exception:
    cap_val = None

out_df = process_df_weight(df, col_name, cap_val=cap_val)

# -------------------- Display --------------------
st.write('### Results (first 200 rows shown)')
display_df = out_df.copy()
if not show_debug:
    for c in ['reductions_pp', 'reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])
st.dataframe(display_df.head(200), use_container_width=True)

# -------------------- Download --------------------
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
    file_name='results_weight_loss.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

st.markdown('---')
st.write('**Rules enforced (Weight Loss):**')
st.write('- Sentence must contain a weight term (weight/body weight/weight loss), a **% number**, and a **reduction cue** (includes “lost”).')
st.write('- Only % values **near** a weight mention are kept (prevents picking up A1c %).')
st.write('- For from→to, the extracted value shown is the **delta** (X − Y).')
st.write('- Optional MAX % cap can be set in the sidebar (blank = no cap).')
