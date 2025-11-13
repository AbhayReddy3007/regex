# streamlit_hba1c_weight_llm.py
import re
import math
import json
import unicodedata
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
API_KEY = "PASTE_YOUR_GEMINI_KEY_HERE"
# =========================================================================

# Lazy import so the app still runs without the package installed
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (sentence→LLM extraction)")

# -------------------- Regex helpers (unchanged logic) --------------------
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

def normalize_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    # Normalize unicode, remove weird replacement artifacts like 'Â' that appear with bad encodings
    t = unicodedata.normalize('NFKC', t)
    # remove stray control characters
    t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', t)
    # fix common mojibake sequences (e.g., "Â·" -> "·")
    t = t.replace('Â·', '·').replace('\ufffd', '')  # replacement char
    return t.strip()

def split_sentences(text: str):
    if not isinstance(text, str):
        return []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(?<=[\.!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    return bool(term_re.search(sent)) and bool(re_pct.search(sent)) and bool(re_reduction_cue.search(sent))

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

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
        'raw': m.group(0),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (abs_start + m.start(), abs_start + m.end()),
    })

# -------------------- Core extraction (same rules) --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red)
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

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
drug_col   = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)
use_llm   = st.sidebar.checkbox('Enable Gemini LLM extraction (Gemini 2.0 Flash)', value=True)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.csv with column named "abstract".')
    st.stop()

# read file robustly with utf-8 fallback & normalization
try:
    if uploaded.name.lower().endswith('.csv'):
        # try utf-8 first, fallback to latin1 to avoid decode errors
        try:
            df = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding='latin1')
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else 0)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

st.success(f'Loaded {len(df)} rows. Processing...')

# -------------------- Gemini setup --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

model = configure_gemini(API_KEY) if use_llm else None

# Norm helper for LLM outputs
def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    # ensure dot as decimal separator
    v = v.replace(',', '.').replace('·', '.')
    return v

# LLM prompt & extractor
LLM_RULES = (
    "You are an extractor. Read the provided SENTENCE(S) and do ONLY the following:\n"
    " - TARGET parameter will be either 'HbA1c' or 'Body weight'. Extract percentage changes that represent\n"
    "   the within-group change (e.g., '13.88% body weight reduction') rather than between-group differences\n"
    "   which are typically noted as 'between-group difference: -13.46%'. Prefer within-group values unless the\n"
    "   sentence explicitly says the primary reported value is the between-group number.\n"
    " - If the sentence contains multiple drugs or groups, use DRUG_NAME parameter (if provided) to pick the value for that drug/group.\n"
    " - Timepoints: prefer 12-month/T12 values over 6-month/T6 values if both are present.\n"
    " - Return STRICT JSON only: {\"extracted\": [\"1.23%\",\"0.85%\"], \"selected_percent\":\"1.23%\"}.\n"
)

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name: str = None):
    if model is None or not sentence or not sentence.strip():
        return [], ""
    prompt = (
        f"TARGET: {target_label}\n"
        f"DRUG_NAME: {drug_name or ''}\n\n"
        f"{LLM_RULES}\n\n"
        f"SENTENCE(S):\n{sentence}\n\n"
        "Return JSON now."
    )
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # find JSON object in response
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = [_norm_percent(x) for x in (data.get("extracted") or []) if isinstance(x, str)]
            selected = _norm_percent(data.get("selected_percent", "") or "")
            return extracted, selected
    except Exception:
        pass
    return [], ""

# -------------------- Scoring functions --------------------
def a1c_score_from_pct(pct_float):
    if pct_float is None or math.isnan(pct_float):
        return ""
    v = abs(pct_float)
    if v > 2.2:
        return 5
    if 1.8 <= v <= 2.2:
        return 4
    if 1.2 <= v <= 1.7:
        return 3
    if 0.8 <= v <= 1.1:
        return 2
    if v < 0.8:
        return 1
    return ""

def weight_score_from_pct(pct_float):
    if pct_float is None or math.isnan(pct_float):
        return ""
    v = abs(pct_float)
    if v >= 22.0:
        return 5
    if 16.0 <= v <= 21.9:
        return 4
    if 10.0 <= v <= 15.9:
        return 3
    if 5.0 <= v <= 9.9:
        return 2
    if v < 5.0:
        return 1
    return ""

# -------------------- Processing (cached) --------------------
@st.cache_data
def process_df(_model, df_in: pd.DataFrame, text_col: str, drug_col: str):
    rows = []
    for _, row in df_in.iterrows():
        raw_text = row.get(text_col, '') or ''
        raw_text = normalize_text(str(raw_text))

        # regex extraction
        hba_matches, hba_sentences = extract_sentences(raw_text, re_hba1c, 'hba1c')
        # enforce HbA1c < 7 rule for single-values (but keep from-to)
        def allowed_hba(m):
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                return float(rp) < 7.0
            for v in (m.get('values') or []):
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) < 7.0:
                        return True
                except:
                    pass
            return False
        hba_matches = [m for m in hba_matches if allowed_hba(m)]
        hba_extracted = [fmt_pct(m.get('reduction_pp')) if 'from-to' in (m.get('type') or '') else m.get('raw') for m in hba_matches]

        wt_matches, wt_sentences = extract_sentences(raw_text, re_weight, 'weight')
        wt_extracted = [fmt_pct(m.get('reduction_pp')) if 'from-to' in (m.get('type') or '') else m.get('raw') for m in wt_matches]

        # sentence strings to feed LLM
        hba_sentence = " | ".join(hba_sentences) if hba_sentences else ""
        wt_sentence  = " | ".join(wt_sentences) if wt_sentences else ""

        # drug_name context if available
        drug_name_val = row.get(drug_col, '') if drug_col in df_in.columns else ''

        # LLM extraction (reads sentence column and returns list + selected string)
        hba_llm_extracted, hba_selected = ([], "")
        wt_llm_extracted, wt_selected = ([], "")

        if _model is not None:
            if hba_sentence:
                hba_llm_extracted, hba_selected = llm_extract_from_sentence(_model, "HbA1c", hba_sentence, drug_name_val)
            if wt_sentence:
                wt_llm_extracted, wt_selected = llm_extract_from_sentence(_model, "Body weight", wt_sentence, drug_name_val)

        # ensure selected % is normalized and positive (abs)
        def selected_to_positive_num(s):
            if not s:
                return None
            s_norm = s.replace('%', '').replace(',', '.').strip()
            try:
                v = float(s_norm)
                return abs(v)
            except:
                return None

        hba_selected_num = selected_to_positive_num(hba_selected)
        wt_selected_num = selected_to_positive_num(wt_selected)

        # scoring
        a1c_score = a1c_score_from_pct(hba_selected_num) if hba_selected_num is not None else ""
        weight_score = weight_score_from_pct(wt_selected_num) if wt_selected_num is not None else ""

        new = row.to_dict()
        # add regex columns
        new.update({
            'sentence': hba_sentence,
            'extracted_matches': hba_extracted,
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
            'weight_sentence': wt_sentence,
            'weight_extracted_matches': wt_extracted,
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
        })
        # add LLM columns
        new.update({
            'LLM extracted': hba_llm_extracted,
            'selected %': (f"{hba_selected_num}%" if hba_selected_num is not None else ""),
            'A1c Score': a1c_score,
            'Weight LLM extracted': wt_llm_extracted,
            'Weight selected %': (f"{wt_selected_num}%" if wt_selected_num is not None else ""),
            'Weight Score': weight_score,
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # keep rows if either target qualifies (as earlier)
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

# run processing (note: pass model as _model to avoid Streamlit hashing error)
out_df = process_df(model, df, col_name, drug_col)

# Put LLM columns next to regex columns
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

# hide debug if requested
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

st.write("### Results (first 200 rows shown)")
st.dataframe(display_df.head(200))

# counts display
counts = out_df.attrs.get('counts', None)
if counts:
    kept, total = counts['kept'], counts['total']
    st.caption(
        f"Kept {kept} of {total} rows ({(kept/total if total else 0):.1%}).  "
        f"HbA1c-only: {counts['hba_only']}, Weight-only: {counts['wt_only']}, Both: {counts['both']}"
    )

# download
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

