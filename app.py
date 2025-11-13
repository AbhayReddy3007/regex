# streamlit_hba1c_weight_llm.py
import re
import math
import json
import unicodedata
from io import BytesIO
from typing import Tuple, List

import pandas as pd
import streamlit as st

# ===================== HARD-CODED GEMINI 2.0 FLASH KEY (as requested) =====================
# NOTE: keep this file local; do NOT commit to source control if this is a real key.
API_KEY = "WEJKEBHABLRKJVBR;KEARVBBVEKJ"
# ========================================================================================

# Lazy import so the app runs if package missing
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (reads sentence column)")

# -------------------- Regex helpers (original logic) --------------------
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

def fix_encoding_artifacts(text: str) -> str:
    """Normalize unicode and replace common mojibake sequences with sane chars."""
    if not isinstance(text, str):
        return text
    # Common replacements found in PDFs -> CSVs
    repl = {
        '\u00c2\u00b7': '·',  # Â· -> middle dot
        '\u00c2': '',         # stray Â
        '\u00e2\u0082\u00ac\u00a0': ' ',  # weird
        'Ã¢': '’',
        'â‰¥': '≥',
        'â‰¤': '≤',
        'â€“': '–',
        'â€”': '—',
        'Ã©': 'é',
        '\ufffd': '',  # replacement char
    }
    # apply replacements (keep it small)
    for k, v in repl.items():
        text = text.replace(k, v)
    # Unicode normalize and strip
    text = unicodedata.normalize('NFKC', text)
    return text

def split_sentences(text: str):
    """Conservative sentence splitter on ., ?, ! or newlines."""
    if not isinstance(text, str):
        return []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(?<=[\.!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    """Require: target term AND a % AND a reduction cue."""
    return bool(term_re.search(sent)) and bool(re_pct.search(sent)) and bool(re_reduction_cue.search(sent))

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

def normalize_percent_str(s: str) -> str:
    """Return a cleaned percent string like '1.23%' or empty."""
    if not s:
        return ''
    s = str(s).strip()
    s = s.replace(' ', '').replace('·', '.').replace(',', '.')
    m = re.search(r'[+-]?\d+(?:\.\d+)?', s)
    if not m:
        return ''
    val = m.group(0)
    return f"{abs(float(val)):.3f}".rstrip('0').rstrip('.') + '%'

# --- build a local window by counting spaces (and INCLUDE bordering tokens) ---
def window_prev_next_spaces_inclusive_tokens(s: str, pos: int, n_prev_spaces: int = 5, n_next_spaces: int = 5):
    space_like = set([' ', '\t', '\n', '\r'])
    L = len(s)

    # Left
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

    # Right
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

    # Clamp
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

# -------------------- Core extraction --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []

    # 1) WHOLE-SENTENCE: from X% to Y%  -> X - Y (pp)
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red)

    # 2) ±5-SPACES window (inclusive) around each target term: other patterns only
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

    # 3) Weight safety: nearest previous % within 60 chars if no window hit
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

    # de-dupe by span
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
    """Return (matches, sentences_used) for sentences meeting the criterion."""
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

# -------------------- Gemini helpers --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

LLM_SYSTEM = (
    "You are an extractor: read the provided SENTENCE(S) and extract ONLY percentage changes for the TARGET.\n"
    "- TARGET is either 'HbA1c' (or A1c) or 'Body weight'.\n"
    "- Extract only values that refer to the TARGET. Do NOT extract thresholds or criteria (e.g., 'HbA1c < 7.0%') — those are NOT reductions.\n"
    "- Prefer within-group reported percent change (e.g., 'yielded a 13.88% body weight reduction') rather than between-group difference, **unless** the sentence explicitly only reports between-group difference and the within-group values are absent.\n"
    "- If multiple drugs/arms are mentioned, use the provided DRUG_NAME context (if given) to pick values associated with that drug.\n"
    "- Prefer the later timepoint when T12/12 months vs T6/6 months both appear (T12 > T6).\n"
    "- If 'from X% to Y%' appears for the target, compute X - Y (percentage points) and report it as the primary reduction.\n"
    "- Return STRICT JSON only, no commentary, EXACT format:\n"
    "{\"extracted\": [\"1.23%\", \"0.85%\"], \"selected_percent\": \"1.23%\"}\n"
    "If nothing relevant, return {\"extracted\": [], \"selected_percent\": \"\"}.\n"
)

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name: str = None) -> Tuple[List[str], str]:
    """Ask Gemini to extract percentages for the target from the sentence string.
       Returns (extracted_list_of_percent_strings, selected_percent_string).
    """
    sentence = fix_encoding_artifacts(sentence or "")
    if not model or not sentence.strip():
        return [], ""
    prompt = (
        f"TASK: Extract percentages for TARGET.\n"
        f"TARGET: {target_label}\n"
        f"{'DRUG_NAME: ' + drug_name + '\\n' if drug_name else ''}"
        f"{LLM_SYSTEM}\n"
        f"SENTENCE(S):\n{sentence}\n"
    )
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # find JSON blob
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e > s:
            js = text[s:e+1]
            data = json.loads(js)
            extracted = [normalize_percent_str(x) for x in (data.get("extracted") or []) if isinstance(x, str)]
            selected = normalize_percent_str(data.get("selected_percent", "") or "")
            return extracted, selected
    except Exception:
        # swallow errors and fallback
        pass
    return [], ""

# -------------------- Scoring --------------------
def a1c_score_from_percent(pct_float: float) -> int:
    # Interpret A1c selected% numeric (assumed positive)
    # Ranges inferred from user text:
    # 5: >=2.2%
    # 4: 1.8 - 2.19%
    # 3: 1.2 - 1.79%
    # 2: 0.8 - 1.19%
    # 1: <0.8%
    if pct_float is None or math.isnan(pct_float):
        return 0
    if pct_float >= 2.2:
        return 5
    if 1.8 <= pct_float < 2.2:
        return 4
    if 1.2 <= pct_float < 1.8:
        return 3
    if 0.8 <= pct_float < 1.2:
        return 2
    if pct_float < 0.8:
        return 1
    return 0

def weight_score_from_percent(pct_float: float) -> int:
    # Weights:
    # 5: >=22%
    # 4: 16-21.9%
    # 3: 10-15.9%
    # 2: 5-9.9%
    # 1: <5%
    if pct_float is None or math.isnan(pct_float):
        return 0
    if pct_float >= 22:
        return 5
    if 16 <= pct_float < 22:
        return 4
    if 10 <= pct_float < 16:
        return 3
    if 5 <= pct_float < 10:
        return 2
    if pct_float < 5:
        return 1
    return 0

# -------------------- Main processing (do NOT hash model param) --------------------
@st.cache_data
def process_df(_model, df: pd.DataFrame, text_col: str, drug_col: str = None) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        # read text, clean encoding artifacts
        raw_text = row.get(text_col, "") or ""
        raw_text = fix_encoding_artifacts(str(raw_text))
        # Extract regex-based sentences & matches (these functions search entire abstract and return only qualifying sentences)
        hba_matches, hba_sentences = extract_sentences(raw_text, re_hba1c, 'hba1c')
        wt_matches, wt_sentences = extract_sentences(raw_text, re_weight, 'weight')

        # apply HbA1c numeric filter (strict <7 on found value or delta allowed)
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

        # Format regex extracted strings
        def fmt_extracted(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        hba_extracted = [fmt_extracted(m) for m in hba_matches]
        wt_extracted  = [fmt_extracted(m) for m in wt_matches]

        # Prepare sentences string (joined) for LLM consumption
        hba_sentence = " | ".join(hba_sentences) if hba_sentences else ""
        wt_sentence  = " | ".join(wt_sentences)  if wt_sentences  else ""

        # LLM extraction (use drug_name context if present)
        drug_name = None
        if drug_col and drug_col in row and pd.notna(row[drug_col]):
            drug_name = str(row[drug_col])

        # HbA1c LLM
        hba_llm_extracted, hba_selected = [], ""
        if _model is not None and hba_sentence:
            try:
                hba_llm_extracted, hba_selected = llm_extract_from_sentence(_model, "HbA1c", hba_sentence, drug_name)
            except Exception:
                hba_llm_extracted, hba_selected = [], ""

        # Weight LLM
        wt_llm_extracted, wt_selected = [], ""
        if _model is not None and wt_sentence:
            try:
                wt_llm_extracted, wt_selected = llm_extract_from_sentence(_model, "Body weight", wt_sentence, drug_name)
            except Exception:
                wt_llm_extracted, wt_selected = [], ""

        # Normalize selected to positive and numeric
        def sel_to_float_percent(sel_str: str):
            s = normalize_percent_str(sel_str)
            if not s:
                return None
            try:
                return float(s.replace('%', ''))
            except:
                return None

        hba_sel_val = sel_to_float_percent(hba_selected)
        wt_sel_val  = sel_to_float_percent(wt_selected)

        # If LLM didn't produce but regex did, fallback to first regex (prefer from-to delta if present)
        if not hba_sel_val and hba_extracted:
            # pick first that looks like a delta (contains '-'? from-to) or first numeric
            for cand in hba_extracted:
                n = sel_to_float_percent(cand)
                if n is not None:
                    hba_sel_val = n
                    break

        if not wt_sel_val and wt_extracted:
            for cand in wt_extracted:
                n = sel_to_float_percent(cand)
                if n is not None:
                    wt_sel_val = n
                    break

        # Scores
        a1c_score = a1c_score_from_percent(hba_sel_val) if hba_sel_val is not None else 0
        weight_score = weight_score_from_percent(wt_sel_val) if wt_sel_val is not None else 0

        # Build row
        new = row.to_dict()
        new.update({
            'sentence': hba_sentence,
            'extracted_matches': hba_extracted,
            'LLM extracted': hba_llm_extracted,
            'selected %': (f"{abs(hba_sel_val):.3f}".rstrip('0').rstrip('.') + '%' if hba_sel_val is not None else ''),
            'A1c Score': a1c_score,

            'weight_sentence': wt_sentence,
            'weight_extracted_matches': wt_extracted,
            'Weight LLM extracted': wt_llm_extracted,
            'Weight selected %': (f"{abs(wt_sel_val):.3f}".rstrip('0').rstrip('.') + '%' if wt_sel_val is not None else ''),
            'Weight Score': weight_score,
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # Keep rows if HbA1c or Weight qualifies
    def _list_has_items(x):
        return isinstance(x, list) and len(x) > 0
    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(_list_has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(_list_has_items))
    mask_keep = mask_hba | mask_wt
    out = out[mask_keep].reset_index(drop=True)

    # counts
    out.attrs['counts'] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum()),
    )
    return out

# -------------------- Streamlit UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
drug_col   = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)
use_llm    = st.sidebar.checkbox('Enable Gemini 2.0 Flash LLM', value=True)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar.')
    st.stop()

# Read file (tolerant of encoding)
try:
    if uploaded.name.lower().endswith('.csv'):
        # attempt utf-8, fallback to latin-1 to avoid encoding errors
        try:
            df = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            df = pd.read_csv(uploaded, encoding='latin1')
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else None)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if isinstance(df, dict):
    st.error("Uploaded file could not be parsed into a single dataframe. Check your file.")
    st.stop()

if col_name not in df.columns:
    st.error(f'Column \"{col_name}\" not found. Available columns: {list(df.columns)}')
    st.stop()

# Configure Gemini (hard-coded key)
model = configure_gemini(API_KEY) if (use_llm and GENAI_AVAILABLE) else None
if use_llm and not GENAI_AVAILABLE:
    st.sidebar.warning("google.generativeai not installed or not available; LLM columns will be empty.")
if use_llm and model is None and GENAI_AVAILABLE:
    st.sidebar.warning("Gemini model not configured (check API key).")

# Process (note: first argument _model avoids hashing the model object)
out_df = process_df(model, df, col_name, drug_col if drug_col in df.columns else None)

# Reorder columns so LLM columns are adjacent to regex columns
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

# Hide debug if requested
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

# Download
@st.cache_data
def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

st.download_button(
    'Download results as Excel',
    data=to_excel_bytes(display_df),
    file_name='results_with_llm_from_sentence.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
