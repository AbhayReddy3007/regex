# streamlit_hba1c_weight_llm.py
import re
import math
import json
from io import BytesIO
from typing import Tuple, List, Optional

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
# Replace the empty string with your real Gemini API key (Gemini 2.0 Flash).
API_KEY = ""   # <-- PASTE YOUR GEMINI 2.0 FLASH KEY HERE
# =========================================================================

# Lazy import for Gemini so app still runs without the package
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (reads sentence columns)")

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
re_weight = re.compile(r'\b(body\s*weight|weight|bw)\b', FLAGS)
re_reduction_cue = re.compile(
    r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b',
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
    parts = re.split(r'(?<=[\.!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    return bool(term_re.search(sent)) and bool(re_pct.search(sent)) and bool(re_reduction_cue.search(sent))

def fmt_pct_num(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

def norm_percent_str(v: str) -> str:
    if not v:
        return ''
    v = str(v).strip().replace(' ', '').replace('·', '.').replace(',', '.')
    # strip trailing punctuation
    v = re.sub(r'[^\d\.\-\+%]+$', '', v)
    if v.endswith('%'):
        num = v[:-1]
    else:
        num = v
    if re.match(r'^[+-]?\d+(?:\.\d+)?$', num):
        # normalize to positive/negative number string with % sign
        return f"{num}%"
    return v

def percent_to_float(v: str) -> Optional[float]:
    if not v:
        return None
    s = str(v).replace('%', '').replace(' ', '').replace(',', '.')
    try:
        return float(s)
    except Exception:
        return None

# --- window by spaces (previous/next n) ---
def window_prev_next_spaces_inclusive_tokens(s: str, pos: int, n_prev_spaces: int = 5, n_next_spaces: int = 5):
    space_like = set([' ', '\t', '\n', '\r'])
    L = len(s)

    # Left side
    i = pos - 1
    spaces = 0
    left_boundary_start = pos  # default
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

def add_match(out_list, si, abs_start, m, typ, values, reduction):
    out_list.append({
        'raw': m.group(0),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (abs_start + m.start(), abs_start + m.end()),
    })

# -------------------- Core extraction --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []

    # 1) Whole sentence 'from X% to Y%'
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red)

    # 2) ±5-spaces window
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

    # 3) Weight fallback: nearest previous % within 60 chars
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

def extract_sentences(text: str, term_re: re.Pattern, tag_prefix: str):
    matches, sentences_used = [], []
    for si, sent in enumerate(split_sentences(text)):
        if not sentence_meets_criterion(sent, term_re):
            continue
        sentences_used.append(sent)
        matches.extend(extract_in_sentence(sent, si, term_re, tag_prefix))

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

# -------------------- LLM (Gemini 2.0 Flash) helpers --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

LLM_RULES = (
    "You are an information extraction assistant. Read the provided SENTENCE(S) and:\n"
    "1) Extract ONLY percentage changes for the TARGET (either HbA1c/A1c or Body weight).\n"
    "2) If multiple percentages appear at different timepoints, prefer 12 months / T12 over 6 months / T6.\n"
    "   Keywords: '12 months', '12-mo', 'T12' > '6 months', '6-mo', 'T6'.\n"
    "3) If 'from X% to Y%' appears, the reported change is (X - Y) percentage points.\n"
    "4) Return STRICT JSON only: {\"extracted\": [\"1.23%\",\"0.85%\"], \"selected_percent\": \"1.23%\"}.\n"
    "5) Do not include non-target values (for HbA1c target ignore weight % and vice versa).\n"
)

def llm_extract_from_sentence(model, target_label: str, sentence: str) -> Tuple[List[str], str]:
    """Call Gemini; return (extracted_list_of_strings, selected_percent_string)"""
    if model is None or not sentence.strip():
        return [], ""
    prompt = (
        f"TARGET: {target_label}\n\n"
        f"{LLM_RULES}\n\n"
        f"SENTENCE(S):\n{sentence}\n\n"
        "Return JSON only."
    )
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted_raw = data.get("extracted") or []
            extracted = []
            for v in extracted_raw:
                if isinstance(v, str):
                    extracted.append(norm_percent_str(v))
            selected = norm_percent_str(data.get("selected_percent", "") or "")
            return extracted, selected
    except Exception:
        pass
    # fallback -> empty
    return [], ""

# -------------------- Scoring functions --------------------
def score_a1c(selected_percent_float: Optional[float]) -> Optional[int]:
    if selected_percent_float is None:
        return None
    v = abs(selected_percent_float)
    # bins per user:
    # 5: >2.2%
    # 4: 1.8%-2.1%
    # 3: 1.2%-1.7%
    # 2: 0.8%-1.1%
    # 1: <0.8%
    if v > 2.2:
        return 5
    if 1.8 <= v <= 2.1:
        return 4
    if 1.2 <= v <= 1.7:
        return 3
    if 0.8 <= v <= 1.1:
        return 2
    if v < 0.8:
        return 1
    # edge cases
    return 1

def score_weight(selected_percent_float: Optional[float]) -> Optional[int]:
    if selected_percent_float is None:
        return None
    v = abs(selected_percent_float)
    # weight bins:
    # 5: >=22%
    # 4: 16-21.9%
    # 3: 10-15.9%
    # 2: 5-9.9%
    # 1: <5%
    if v >= 22:
        return 5
    if 16 <= v <= 21.9:
        return 4
    if 10 <= v <= 15.9:
        return 3
    if 5 <= v <= 9.9:
        return 2
    if v < 5:
        return 1
    return 1

# -------------------- UI: upload + sheet + column selection --------------------
st.sidebar.header('Options')
uploaded = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
sheet_choice = None
if uploaded and uploaded.name.lower().endswith(('.xls', '.xlsx')):
    # read sheets to let user pick
    try:
        xls = pd.read_excel(uploaded, sheet_name=None)
        sheet_names = list(xls.keys())
        sheet_choice = st.sidebar.selectbox("Select sheet", options=sheet_names, index=0)
        df = xls[sheet_choice]
    except Exception as e:
        st.sidebar.error(f"Failed to read Excel: {e}")
        st.stop()
elif uploaded and uploaded.name.lower().endswith('.csv'):
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        st.stop()
else:
    df = None

if df is None:
    st.info("Upload a CSV or Excel file (supports multiple columns).")
    st.stop()

# let user pick the text column (abstract) from all columns
col_name = st.sidebar.selectbox("Select column containing abstracts/text", options=list(df.columns), index=0)
st.sidebar.markdown("---")
use_llm = st.sidebar.checkbox("Enable Gemini 2.0 Flash LLM extraction (must hard-code API_KEY)", value=True)
if use_llm and not API_KEY:
    st.sidebar.warning("Gemini API key (API_KEY) is empty. Edit the script and paste your key into API_KEY variable.")

sheet_info = f" (sheet: {sheet_choice})" if sheet_choice else ""
st.success(f"Loaded {len(df)} rows{sheet_info}. Processing...")

# -------------------- Processing (regex extraction + LLM on sentence columns) --------------------
@st.cache_data
def process_and_annotate(df: pd.DataFrame, text_col: str, use_llm_flag: bool):
    model = configure_gemini(API_KEY) if use_llm_flag else None

    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '') or ''
        text = '' if not isinstance(text, str) else text

        # HbA1c regex
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')
        # keep only per your rule: allowed hba <7 or from-to allowed
        def allowed_hba(m):
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                return float(abs(rp)) < 7.0
            for v in (m.get('values') or []):
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and float(abs(v)) < 7.0:
                        return True
                except Exception:
                    pass
            return False

        hba_matches = [m for m in hba_matches if allowed_hba(m)]
        hba_extracted_regex = []
        for m in hba_matches:
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                hba_extracted_regex.append(fmt_pct_num(m.get('reduction_pp')))
            else:
                hba_extracted_regex.append(m.get('raw', ''))

        # Weight regex
        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')
        wt_extracted_regex = []
        for m in wt_matches:
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                wt_extracted_regex.append(fmt_pct_num(m.get('reduction_pp')))
            else:
                wt_extracted_regex.append(m.get('raw', ''))

        # Build sentence columns (join multiple qualifying sentences by ' | ')
        sentence_joined = ' | '.join(hba_sentences) if hba_sentences else ''
        weight_sentence_joined = ' | '.join(wt_sentences) if wt_sentences else ''

        # LLM: call only if model available and sentence exists; else empty lists
        hba_llm_extracted, hba_selected = ([], "")
        if model is not None and sentence_joined:
            raw_extracted, raw_selected = llm_extract_from_sentence(model, "HbA1c", sentence_joined)
            # choose normalized values
            hba_llm_extracted = [norm_percent_str(x) for x in raw_extracted]
            hba_selected = norm_percent_str(raw_selected) if raw_selected else ""

        wt_llm_extracted, wt_selected = ([], "")
        if model is not None and weight_sentence_joined:
            raw_extracted_w, raw_selected_w = llm_extract_from_sentence(model, "Body weight", weight_sentence_joined)
            wt_llm_extracted = [norm_percent_str(x) for x in raw_extracted_w]
            wt_selected = norm_percent_str(raw_selected_w) if raw_selected_w else ""

        # If LLM missing or returned empty, fallback: use regex candidates (prefer first)
        if not hba_llm_extracted and hba_extracted_regex:
            hba_llm_extracted = [norm_percent_str(x) for x in hba_extracted_regex]
        if not hba_selected and hba_llm_extracted:
            hba_selected = hba_llm_extracted[0]

        if not wt_llm_extracted and wt_extracted_regex:
            wt_llm_extracted = [norm_percent_str(x) for x in wt_extracted_regex]
        if not wt_selected and wt_llm_extracted:
            wt_selected = wt_llm_extracted[0]

        # Ensure selected % is positive numeric if possible
        def make_positive_selected(s):
            if not s:
                return ""
            f = percent_to_float(s)
            if f is None:
                return s
            return f"{abs(f)}%"

        hba_selected_pos = make_positive_selected(hba_selected)
        wt_selected_pos = make_positive_selected(wt_selected)

        # compute numeric for scoring
        hba_selected_num = percent_to_float(hba_selected_pos)
        wt_selected_num = percent_to_float(wt_selected_pos)

        hba_score = score_a1c(hba_selected_num)
        wt_score = score_weight(wt_selected_num)

        newrow = row.to_dict()
        # add columns
        newrow.update({
            'sentence': sentence_joined,
            'extracted_matches': hba_extracted_regex,
            'LLM extracted': hba_llm_extracted,
            'selected %': hba_selected_pos,
            'A1c Score': hba_score,

            'weight_sentence': weight_sentence_joined,
            'weight_extracted_matches': wt_extracted_regex,
            'Weight LLM extracted': wt_llm_extracted,
            'Weight selected %': wt_selected_pos,
            'Weight Score': wt_score,
        })
        rows.append(newrow)

    out = pd.DataFrame(rows)

    # keep rows where either HbA1c or weight has extracted values
    def _list_has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(_list_has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(_list_has_items))
    mask_keep = mask_hba | mask_wt
    out = out[mask_keep].reset_index(drop=True)

    # store counts
    out.attrs['counts'] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum()),
    )
    return out

# run processing
out_df = process_and_annotate(df, col_name, use_llm)

# -------------------- Reorder columns: put LLM columns beside regex columns --------------------
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
cols = list(display_df.columns)
cols = insert_after(cols, "extracted_matches", ["LLM extracted", "selected %", "A1c Score"])
cols = insert_after(cols, "weight_extracted_matches", ["Weight LLM extracted", "Weight selected %", "Weight Score"])
display_df = display_df[cols]

# optionally hide debug columns
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

st.write("### Results (first 200 rows shown)")
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
def to_excel_bytes(df_):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df_.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(display_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_llm_and_scores.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
