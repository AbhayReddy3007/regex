# streamlit_hba1c_weight_llm.py
import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
# Replace the empty string below with your real key.
API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"
# =========================================================================

# Lazy Gemini import so the app still runs if the package isn't available
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (sentence + drug_name aware)")

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

# -------------------- Utilities (your original + extras) --------------------
def parse_number(s: str) -> float:
    if s is None:
        return float('nan')
    s = str(s).replace(',', '.').replace('·', '.').strip()
    try:
        return float(s)
    except Exception:
        return float('nan')

def sanitize_text(t: str) -> str:
    """Try to fix common mojibake & weird unicode sequences and trim."""
    if not isinstance(t, str):
        return ''
    # Common artifacts seen in scraped PDFs/CSVs
    t = t.replace('\u00c2\u00b7', '·')  # Â·
    t = t.replace('\u00c2', '')        # Â leftover
    t = t.replace('â‰¥', '>=')
    t = t.replace('â€“', '-')
    t = t.replace('â€", '"')
    t = t.replace('\u2009', ' ')  # narrow no-break space
    # If still odd non-ascii sequences, try a safe replace
    t = t.replace('\xad', '')  # soft hyphen
    return t.strip()

def split_sentences(text: str):
    """Conservative sentence splitter on ., ?, ! or newlines."""
    if not isinstance(text, str):
        return []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(?<=[\.!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    """Require: (target term) AND a % AND a reduction cue."""
    return bool(term_re.search(sent)) and bool(re_pct.search(sent)) and bool(re_reduction_cue.search(sent))

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

# --- build a local window by counting spaces (and INCLUDE bordering tokens) ---
def window_prev_next_spaces_inclusive_tokens(s: str, pos: int, n_prev_spaces: int = 5, n_next_spaces: int = 5):
    space_like = set([' ', '\t', '\n', '\r'])
    L = len(s)

    # --- Left side ---
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

    # --- Right side ---
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

# -------------------- Core extraction (unchanged behavior) --------------------
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

# -------------------- Gemini helpers --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v

LLM_RULES = (
    "You are an information extraction assistant.\n"
    "Read the provided sentence(s) and extract percentage CHANGES for the TARGET only (HbA1c or body weight).\n"
    "If the sentence mentions more than one drug/arm, prefer the value that corresponds to the provided drug_name.\n"
    "If both within-group and between-group values appear, prefer within-group change for the specific drug (unless asked otherwise).\n"
    "If 'from X% to Y%' or 'declined from X% to Y%' appears, compute X - Y (percentage points) as the change.\n"
    "If multiple timepoints appear, prefer 12 months / T12 over 6 months / T6. Keywords: '12 months','12-mo','T12' > '6 months','6-mo','T6'.\n"
    "Return STRICT JSON only: {\"extracted\": [\"1.23%\",\"0.85%\"], \"selected_percent\": \"1.23%\"}.\n"
    "If no valid target percentages, return: {\"extracted\": [], \"selected_percent\": \"\"}.\n"
)

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name: str = ""):
    """Call Gemini with a small prompt that includes drug_name (if present). Returns (list[str], selected_str)."""
    if model is None or not sentence.strip():
        return [], ""
    # include drug_name context to bias selection when multiple drugs mentioned
    drug_part = f"DRUG_NAME: {drug_name}\n" if drug_name else ""
    prompt = (
        f"{drug_part}TARGET: {target_label}\n\n{LLM_RULES}\n\nSENTENCE:\n{sentence}\n\nReturn JSON only.\n"
    )
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = [_norm_percent(x) for x in (data.get("extracted") or []) if isinstance(x, str)]
            selected = _norm_percent(data.get("selected_percent", ""))
            # make selected positive if numeric and negative sign present
            if selected:
                try:
                    num = parse_number(selected.replace('%',''))
                    if not math.isnan(num):
                        selected = fmt_pct(abs(num))
                except Exception:
                    pass
            # ensure extracted are normalized with % sign
            extracted = [ (x if x.endswith('%') else (x + '%')) for x in extracted ]
            return extracted, selected
    except Exception:
        pass
    return [], ""

# -------------------- Scoring rules --------------------
def a1c_score_from_percent_str(s: str) -> int:
    if not s: return None
    try:
        v = parse_number(s.replace('%',''))
    except:
        return None
    if v > 2.2: return 5
    if 1.8 <= v <= 2.1: return 4
    if 1.2 <= v <= 1.7: return 3
    if 0.8 <= v <= 1.1: return 2
    if v < 0.8: return 1
    return None

def weight_score_from_percent_str(s: str) -> int:
    if not s: return None
    try:
        v = parse_number(s.replace('%',''))
    except:
        return None
    if v >= 22: return 5
    if 16 <= v <= 21.9: return 4
    if 10 <= v <= 15.9: return 3
    if 5 <= v <= 9.9: return 2
    if v < 5: return 1
    return None

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
drug_col   = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)
use_llm    = st.sidebar.checkbox('Enable Gemini 2.0 Flash (LLM columns)', value=True)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar.')
    st.stop()

# read file robustly (and sanitize text columns)
try:
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded, dtype=str, encoding='utf-8', low_memory=False)
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else 0, dtype=str)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

# configure model (don't pass model object into cached functions)
model = configure_gemini(API_KEY) if use_llm else None
if use_llm and model is None:
    st.warning("Gemini model not available (check API key and package). LLM columns will be empty.")

# -------------------- Main processing (NOT cached to avoid streamlit hashing issues) --------------------
def process_df(df, text_col, drug_col_name=None, model=None):
    rows = []
    for _, row in df.iterrows():
        raw_text = row.get(text_col, '') or ''
        text = sanitize_text(raw_text)
        # --- regex extraction as before ---
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')

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
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')

        def fmt_extracted_wt(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        # prepare row
        new = row.to_dict()

        # normalize sentences (join) and ensure they include the desired earlier sentence capturing behavior
        joined_hba_sentence = ' | '.join(hba_sentences) if hba_sentences else ''
        joined_wt_sentence = ' | '.join(wt_sentences) if wt_sentences else ''

        new.update({
            'sentence': joined_hba_sentence,
            'extracted_matches': [fmt_extracted_hba(m) for m in hba_matches],
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
            'weight_sentence': joined_wt_sentence,
            'weight_extracted_matches': [fmt_extracted_wt(m) for m in wt_matches],
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
        })

        # LLM extraction: the LLM should read the sentence and drug_name (if available)
        drug_value = (row.get(drug_col_name, '') or '') if drug_col_name and drug_col_name in df.columns else ''
        # HbA1c LLM
        hba_llm_extracted, hba_selected = ([], "")
        if model is not None and joined_hba_sentence:
            hba_llm_extracted, hba_selected = llm_extract_from_sentence(model, "HbA1c", joined_hba_sentence, drug_value)
        # Weight LLM
        wt_llm_extracted, wt_selected = ([], "")
        if model is not None and joined_wt_sentence:
            wt_llm_extracted, wt_selected = llm_extract_from_sentence(model, "Body weight", joined_wt_sentence, drug_value)

        # If LLM didn't extract anything, ensure selected remains empty (do not fallback)
        if not hba_llm_extracted:
            hba_selected = ""
        if not wt_llm_extracted:
            wt_selected = ""

        # Ensure selected % is positive number if present (abs)
        def make_positive_percent(s):
            if not s:
                return ""
            try:
                n = parse_number(s.replace('%',''))
                if math.isnan(n):
                    return ""
                return fmt_pct(abs(n))
            except:
                return s

        hba_selected = make_positive_percent(hba_selected)
        wt_selected = make_positive_percent(wt_selected)

        # Compute scores next to selected % columns
        a1c_score = a1c_score_from_percent_str(hba_selected) if hba_selected else None
        weight_score = weight_score_from_percent_str(wt_selected) if wt_selected else None

        new.update({
            'LLM extracted': hba_llm_extracted,
            'selected %': hba_selected,
            'A1c Score': a1c_score,
            'Weight LLM extracted': wt_llm_extracted,
            'Weight selected %': wt_selected,
            'Weight Score': weight_score
        })

        rows.append(new)

    out = pd.DataFrame(rows)

    # keep rows if hba OR weight qualifies (same rule as before)
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

# Run processing
out_df = process_df(df, col_name, drug_col if drug_col in df.columns else None, model)

# -------------------- Reorder columns: place LLM columns BESIDE regex columns --------------------
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

# Hide debug columns unless requested
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
def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(display_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_llm_and_scores.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
