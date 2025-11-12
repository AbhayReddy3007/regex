# streamlit_hba1c_weight_llm.py
import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
API_KEY = ""   # <- REPLACE with your Gemini API key (gemini-2.0-flash)
# =========================================================================

# Lazy Gemini import so the app still runs without it
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (reads the sentence columns)")

# -------------------- Regex helpers (your original) --------------------
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
re_weight = re.compile(r'\b(body\s*weight|weight|bw)\b', FLAGS)
re_reduction_cue = re.compile(
    r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b',
    FLAGS
)

# -------------------- Utilities (your original) --------------------
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
    parts = re.split(r'(?<=[\.!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    return bool(term_re.search(sent)) and bool(re_pct.search(sent)) and bool(re_reduction_cue.search(sent))

def fmt_pct_val(v):
    # return normalized percent string like '1.2%' or '' if invalid
    if v is None:
        return ''
    try:
        if isinstance(v, str) and v.endswith('%'):
            v = v[:-1]
        f = float(v)
        # strip trailing zeros
        s = f"{abs(f):.3f}".rstrip('0').rstrip('.')
        return f"{s}%"
    except Exception:
        return ''

def window_prev_next_spaces_inclusive_tokens(s: str, pos: int, n_prev_spaces: int = 5, n_next_spaces: int = 5):
    space_like = set([' ', '\t', '\n', '\r'])
    L = len(s)

    # Left side
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

    # Right side
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
        'raw': m.group(0),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (abs_start + m.start(), abs_start + m.end()),
    })

# -------------------- Core extraction (your original) --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []

    # whole sentence from→to
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red)

    # ±5-space window around each term
    any_window_hit = False
    for hh in term_re.finditer(sent):
        seg, abs_s, _ = window_prev_next_spaces_inclusive_tokens(sent, hh.end(), 5, 5)

        for m in re_reduce_by.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:percent_or_pp_pmSpaces5', [v], v)

        for m in re_abs_pp.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:pp_word_pmSpaces5', [v], v)

        for m in re_range.finditer(seg):
            any_window_hit = True
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            rep = None if (math.isnan(a) or math.isnan(b)) else max(a, b)
            add_match(matches, si, abs_s, m, f'{tag_prefix}:range_percent_pmSpaces5', [a, b], rep)

        for m in re_pct.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:percent_pmSpaces5', [v], v)

    # weight safety fallback
    if (tag_prefix == 'weight') and (not any_window_hit):
        for hh in term_re.finditer(sent):
            pos = hh.start()
            left = max(0, pos - 60)
            left_chunk = sent[left:pos]
            last_pct = None
            for m in re_pct.finditer(left_chunk):
                last_pct = m
            if last_pct is not None:
                abs_start = left
                v = parse_number(last_pct.group(1))
                add_match(matches, si, abs_start, last_pct, f'{tag_prefix}:percent_prev60chars', [v], v)

    # de-dupe
    seen = set()
    uniq = []
    for mm in matches:
        if mm['span'] in seen:
            continue
        seen.add(mm['span'])
        uniq.append(mm)
    uniq.sort(key=lambda x: x['span'][0])
    return uniq

def extract_sentences(text: str, term_re: re.Pattern, tag_prefix: str):
    matches, sentences_used = [], []
    for si, sent in enumerate(split_sentences(text)):
        if not sentence_meets_criterion(sent, term_re):
            continue
        sentences_used.append(sent)
        matches.extend(extract_in_sentence(sent, si, term_re, tag_prefix))

    seen, filtered = set(), []
    for mm in matches:
        key = (mm['sentence_index'], mm['span'])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(mm)

    filtered.sort(key=lambda x: (x['sentence_index'], x['span'][0]))
    return filtered, sentences_used

# -------------------- UI (file & column selection) --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
# let user pick column if multiple columns present
example_default_col = 'abstract'
col_name_input = st.sidebar.text_input('Column with abstracts/text (leave blank to choose)', value='')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
    st.stop()

# read file robustly
try:
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else None)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

# choose column: prefer the user text input; if blank, choose default if present else show selectbox
if col_name_input and col_name_input in df.columns:
    text_col = col_name_input
else:
    if example_default_col in df.columns and not col_name_input:
        text_col = example_default_col
    else:
        # ask user to choose from columns
        text_col = st.sidebar.selectbox('Select text column', options=list(df.columns), index=0)

st.success(f'Loaded {len(df)} rows. Using column: "{text_col}"')

# -------------------- Regex-only processing (build sentence column and regex extractions) --------------------
@st.cache_data
def process_df_regex_only(df, text_col):
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        text = '' if not isinstance(text, str) else text

        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')

        # HbA1c strict: only values < 7 (prefer delta if present)
        def allowed_hba(m):
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                return float(rp) < 7.0
            for v in (m.get('values') or []):
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) < 7.0:
                        return True
                except Exception:
                    pass
            return False

        hba_matches = [m for m in hba_matches if allowed_hba(m)]

        def fmt_extracted_hba(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct_val(m.get('reduction_pp'))
            return m.get('raw', '')

        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')

        def fmt_extracted_wt(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct_val(m.get('reduction_pp'))
            return m.get('raw', '')

        new = row.to_dict()
        # HbA1c
        new.update({
            'sentence': ' | '.join(hba_sentences) if hba_sentences else '',
            'extracted_matches': [fmt_extracted_hba(m) for m in hba_matches],
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
        })
        # Weight
        new.update({
            'weight_sentence': ' | '.join(wt_sentences) if wt_sentences else '',
            'weight_extracted_matches': [fmt_extracted_wt(m) for m in wt_matches],
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    def _list_has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(_list_has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(_list_has_items))
    mask_keep = mask_hba | mask_wt
    out = out[mask_keep].reset_index(drop=True)

    out.attrs['counts'] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum()),
    )
    return out

out_df = process_df_regex_only(df, text_col)

# -------------------- Gemini 2.0 Flash setup --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

model = configure_gemini(API_KEY)

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if not v:
        return ""
    # remove surrounding chars
    # allow numbers with optional %; convert comma/dot
    v2 = v.replace(',', '.').replace('·', '.')
    m = re.search(r'[+-]?\d+(?:\.\d+)?', v2)
    if not m:
        return ""
    num = float(m.group(0))
    num = abs(num)  # convert negative to positive for selected % as requested
    # format nicely
    s = f"{num:.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

# Prompt rules:
# - LLM reads just the 'sentence' (or 'weight_sentence') string.
# - Extract only percentages pertaining to the requested target (HbA1c or body weight).
# - Prefer T12/12 months over T6/6 months when both appear.
# - If 'from X% to Y%' appears, the reported change is (X - Y) percentage points (LLM may also output reductions phrased "reduced by Z%").
# - Output strict JSON: {"extracted":["1.23%","0.85%"], "selected_percent":"1.23%"} with percent sign and numeric values.
# - Do not include irrelevant values.
LLM_RULES = (
    "You are an information extraction model. Read the given sentence(s) and:\n"
    "1) Extract ONLY percentage changes for the TARGET (either HbA1c/A1c or body weight).\n"
    "2) If multiple values exist at different timepoints, prefer 12 months or T12 over 6 months or T6.\n"
    "   Keywords to interpret time: '12 months', '12-mo', 'T12' > '6 months', '6-mo', 'T6'.\n"
    "3) If 'from X% to Y%' appears, the reported change is (X - Y) percentage points.\n"
    "4) Return STRICT JSON only: {\"extracted\":[\"1.23%\",\"0.85%\"], \"selected_percent\":\"1.23%\"}.\n"
    "5) Do not include non-target values (e.g., body weight when target is HbA1c, or vice versa).\n"
)

def llm_extract_from_sentence(model, target_label: str, sentence: str):
    if model is None or not sentence.strip():
        return [], ""
    prompt = (
        f"TARGET: {target_label}\n\n"
        f"{LLM_RULES}\n\n"
        f"SENTENCE(S):\n{sentence}\n"
    )
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            payload = text[s:e+1]
            data = json.loads(payload)
            extracted = []
            for x in (data.get("extracted") or []):
                if isinstance(x, str):
                    nx = _norm_percent(x)
                    if nx:
                        extracted.append(nx)
            selected = _norm_percent(data.get("selected_percent", "") or "")
            return extracted, selected
    except Exception:
        pass
    # fallback: try to return percentages found by regex in the sentence (limited)
    found = []
    for m in re_pct.finditer(sentence):
        found.append(_norm_percent(m.group(0)))
    sel = found[0] if found else ""
    return found, sel

# -------------------- Run LLM over the *sentence columns* --------------------
# HbA1c
hba_llm = out_df.get("sentence", pd.Series(dtype=str)).fillna("").astype(str).apply(
    lambda s: llm_extract_from_sentence(model, "HbA1c", s)
)
out_df["LLM extracted"] = [vals for vals, sel in hba_llm]
out_df["selected %"]   = [sel  for vals, sel in hba_llm]

# Weight
wt_llm = out_df.get("weight_sentence", pd.Series(dtype=str)).fillna("").astype(str).apply(
    lambda s: llm_extract_from_sentence(model, "Body weight", s)
)
out_df["Weight LLM extracted"] = [vals for vals, sel in wt_llm]
out_df["Weight selected %"]    = [sel  for vals, sel in wt_llm]

# -------------------- Scoring (A1c Score & Weight Score) --------------------
def a1c_score_from_selected_pct(pct_str: str) -> int:
    """
    Scores for A1c:
    5: > 2.2%
    4: 1.8% - 2.1%
    3: 1.2% - 1.7%
    2: 0.8% - 1.1%
    1: < 0.8%
    """
    if not pct_str:
        return 0
    try:
        num = float(pct_str.replace('%',''))
    except:
        return 0
    if num > 2.2:
        return 5
    if 1.8 <= num <= 2.1:
        return 4
    if 1.2 <= num <= 1.7:
        return 3
    if 0.8 <= num <= 1.1:
        return 2
    if num < 0.8:
        return 1
    # If falls into gap (e.g., 1.15 or 1.75), assign nearest bucket sensibly:
    if num < 1.2:
        return 2
    if num < 1.8:
        return 3
    return 1

def weight_score_from_selected_pct(pct_str: str) -> int:
    """
    Scores for weight:
    5: >= 22%
    4: 16 - 21.9%
    3: 10 - 15.9%
    2: 5 - 9.9%
    1: <5%
    """
    if not pct_str:
        return 0
    try:
        num = float(pct_str.replace('%',''))
    except:
        return 0
    if num >= 22:
        return 5
    if 16 <= num <= 21.9:
        return 4
    if 10 <= num <= 15.9:
        return 3
    if 5 <= num <= 9.9:
        return 2
    if num < 5:
        return 1
    return 0

# Apply scores
out_df["A1c Score"] = out_df["selected %"].fillna("").astype(str).apply(lambda x: a1c_score_from_selected_pct(x))
out_df["Weight Score"] = out_df["Weight selected %"].fillna("").astype(str).apply(lambda x: weight_score_from_selected_pct(x))

# -------------------- Reorder columns: place LLM & Score columns BESIDE regex columns --------------------
def insert_after(cols, after, names):
    if after not in cols:
        return cols
    i = cols.index(after)
    for name in names[::-1]:
        if name in cols:
            cols.remove(name)
        cols.insert(i+1, name)
    return cols

display_df = out_df.copy()
# HbA1c LLM columns next to extracted_matches
cols = list(display_df.columns)
cols = insert_after(cols, "extracted_matches", ["LLM extracted", "selected %", "A1c Score"])
# Weight LLM columns next to weight_extracted_matches
cols = insert_after(cols, "weight_extracted_matches", ["Weight LLM extracted", "Weight selected %", "Weight Score"])
# keep only columns that exist
cols = [c for c in cols if c in display_df.columns]
display_df = display_df[cols]

# -------------------- Show results --------------------
st.write("### Results (first 200 rows shown)")
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

st.dataframe(display_df.head(200))

# counts
counts = out_df.attrs.get('counts', None)
if counts:
    kept, total = counts['kept'], counts['total']
    st.caption(
        f"Kept {kept} of {total} rows ({(kept/total if total else 0):.1%}).  "
        f"HbA1c-only: {counts['hba_only']}, Weight-only: {counts['wt_only']}, Both: {counts['both']}"
    )

# -------------------- Download --------------------
@st.cache_data
def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(display_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_llm_from_sentence_and_scores.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
