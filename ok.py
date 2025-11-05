"""
Streamlit HbA1c / A1c reduction extractor (regex-only)

Save this file as `streamlit_hba1c_extractor.py` and run:
    streamlit run streamlit_hba1c_extractor.py

Features:
- Upload XLSX or CSV
- Pick text column and sheet (for Excel)
- Extract reductions using regex (same logic as the earlier script)
- Show results in an interactive table
- Download results as Excel
"""

import streamlit as st
import pandas as pd
import re
import math
from io import BytesIO

st.set_page_config(page_title="HbA1c Reduction Extractor", layout="wide")
st.title("HbA1c / A1c reduction extractor — regex only (no LLM)")

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


def parse_number(s: str) -> float:
    if s is None:
        return float('nan')
    s = s.replace(',', '.').strip()
    try:
        return float(s)
    except:
        return float('nan')


def extract_hba1c_reductions(text: str):
    """
    Prefer matches that occur on the *same line* as an HbA1c/A1c mention.
    If no such same-line matches are found, fall back to the previous wider search
    but mark those matches as `..._fallback` so you can filter them out if desired.
    """
    if not isinstance(text, str):
        return []
    matches = []
    t = text

    # --- 1) line-based search: only consider lines that contain HbA1c/A1c ---
    lines = re.split(r'[\r\n]+', t)
    for li, line in enumerate(lines):
        if not line.strip():
            continue
        if re_hba.search(line):
            # within this line, search for the common patterns
            for m in re_fromto.finditer(line):
                a = parse_number(m.group(1))
                b = parse_number(m.group(2))
                reduction = None
                if not math.isnan(a) and not math.isnan(b):
                    reduction = a - b
                matches.append({'raw': m.group(0), 'type': 'from-to_same_line', 'values': [a, b], 'reduction_pp': reduction, 'span': (li, m.span())})

            for m in re_reduce_by.finditer(line):
                v = parse_number(m.group(1))
                matches.append({'raw': m.group(0), 'type': 'percent_or_pp_same_line', 'values': [v], 'reduction_pp': v, 'span': (li, m.span())})

            for m in re_abs_pp.finditer(line):
                v = parse_number(m.group(1))
                matches.append({'raw': m.group(0), 'type': 'pp_word_same_line', 'values': [v], 'reduction_pp': v, 'span': (li, m.span())})

            for m in re_range.finditer(line):
                a = parse_number(m.group(1))
                b = parse_number(m.group(2))
                matches.append({'raw': m.group(0), 'type': 'range_percent_same_line', 'values': [a, b], 'reduction_pp': max(a, b) if not math.isnan(a) and not math.isnan(b) else None, 'span': (li, m.span())})

            for m in re_pct.finditer(line):
                v = parse_number(m.group(1))
                matches.append({'raw': m.group(0), 'type': 'percent_same_line', 'values': [v], 'reduction_pp': v, 'span': (li, m.span())})

    # If we found same-line matches, return those (deduped)
    if matches:
        # dedupe by (line, span)
        seen = set()
        filtered = []
        for mm in matches:
            sp = mm['span']
            if sp in seen:
                continue
            seen.add(sp)
            filtered.append(mm)
        return filtered

    # --- 2) fallback: no same-line matches found, run the broader search (previous behavior)
    # We append "_fallback" to the type so you can filter these out if you want to keep only same-line hits.

    # from X% to Y%  -> compute X - Y
    for m in re_fromto.finditer(t):
        a = parse_number(m.group(1))
        b = parse_number(m.group(2))
        reduction = None
        if not math.isnan(a) and not math.isnan(b):
            reduction = a - b
        matches.append({
            'raw': m.group(0),
            'type': 'from-to_fallback',
            'values': [a, b],
            'reduction_pp': reduction,
            'span': m.span()
        })

    # explicit "reduced by X%" or similar
    for m in re_reduce_by.finditer(t):
        v = parse_number(m.group(1))
        matches.append({
            'raw': m.group(0),
            'type': 'percent_or_pp_fallback',
            'values': [v],
            'reduction_pp': v,
            'span': m.span()
        })

    # absolute pp phrasing
    for m in re_abs_pp.finditer(t):
        v = parse_number(m.group(1))
        matches.append({
            'raw': m.group(0),
            'type': 'pp_word_fallback',
            'values': [v],
            'reduction_pp': v,
            'span': m.span()
        })

    # ranges like "1.0-1.5%"
    for m in re_range.finditer(t):
        a = parse_number(m.group(1))
        b = parse_number(m.group(2))
        matches.append({
            'raw': m.group(0),
            'type': 'range_percent_fallback',
            'values': [a, b],
            'reduction_pp': max(a, b) if not math.isnan(a) and not math.isnan(b) else None,
            'span': m.span()
        })

    # percent tokens: restrict to those near any HbA1c word within +/-80 chars (original logic)
    pct_hits = []
    for m in re_pct.finditer(t):
        pct_hits.append((m, parse_number(m.group(1))))

    hba_words = list(re_hba.finditer(t))
    if hba_words:
        for m, val in pct_hits:
            start, end = m.span()
            near = any(abs(start - hh.span()[0]) <= 80 or abs(end - hh.span()[1]) <= 80 for hh in hba_words)
            if near:
                matches.append({
                    'raw': m.group(0),
                    'type': 'percent_near_hba1c_fallback',
                    'values': [val],
                    'reduction_pp': val,
                    'span': m.span()
                })
    else:
        for m, val in pct_hits:
            matches.append({
                'raw': m.group(0),
                'type': 'percent_global_fallback',
                'values': [val],
                'reduction_pp': val,
                'span': m.span()
            })

    # dedupe by span (original behavior)
    seen_spans = set()
    filtered = []
    for mm in matches:
        sp = mm['span']
        if sp in seen_spans:
            continue
        seen_spans.add(sp)
        filtered.append(mm)

    filtered.sort(key=lambda x: x['span'][0] if isinstance(x['span'][0], int) else x['span'][0])
    return filtered

def choose_best_reduction(matches):
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
        # Keep ONLY matches that explicitly contain a % sign and are not global (must be associated with HbA1c/A1c context)
        matches = [m for m in matches if '%' in m.get('raw', '') and 'global' not in m.get('type', '')]

        # Additional rule:
        # - Keep numbers < 7
        # - Allow numbers >= 7 ONLY if it's an explicit from-to reduction (e.g., "from 9.1% to 7.8%")
        def _allowed(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:  # allow explicit from-to phrases regardless of value
                return True
            # Otherwise, any numeric value < 7 passes
            vals = m.get('values') or []
            for v in vals:
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) < 7.0:
                        return True
                except Exception:
                    pass
            # Also permit if computed reduction_pp < 7 (for safety)
            rp = m.get('reduction_pp')
            try:
                if rp is not None and not (isinstance(rp, float) and math.isnan(rp)) and float(rp) < 7.0:
                    return True
            except Exception:
                pass
            return False

        matches = [m for m in matches if _allowed(m)]

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
    # drop rows where no extracted matches
    out_df = out_df[out_df['extracted_matches'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    # drop rows where best_reduction_pp is NaN
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
st.write('- The extractor uses regex; it normalizes to *absolute percentage points* where possible (e.g., "from 9.0% to 7.5%" -> 1.5).')
st.write('- Phrases like "reduced by 20%" are captured as numeric 20 (check `reduction_types` to see if it was percent vs pp).')
st.write('- If you want the app to interpret relative % vs absolute pp differently, tell me and I will update the logic.')

st.write('')
st.write('Made for you — drop your file above and download the cleaned results.')
