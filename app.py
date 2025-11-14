# streamlit_hba1c_weight_llm.py
import re
import math
import json
import unicodedata
from io import BytesIO

import pandas as pd
import streamlit as st

# -------------------- HARD-CODED GEMINI KEY (REPLACE THIS) --------------------
# I cannot provide a real API key for security reasons. Replace the string below
# with your own Gemini API key before running the app.
API_KEY = "<REPLACE_WITH_YOUR_KEY>"
# -----------------------------------------------------------------------------

# Lazy import of google.generativeai so app still runs if library is missing
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (sentence-based LLM extraction)")

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

def normalize_text(t: str) -> str:
    """Normalize encoding artifacts and Unicode marks, make nicer for LLM."""
    if not isinstance(t, str):
        return ''
    # fix encoding artifacts and normalize
    t = unicodedata.normalize('NFKC', t)
    # replace non-breaking spaces, weird bullets etc.
    t = t.replace('\u00a0', ' ').replace('\u200b', '')
    return t

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

def fmt_pct_val(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

# window by counting spaces, include border tokens
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

# -------------------- Core extraction (regex) --------------------
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
        # user asked for gemini 2.0 flash
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v

# LLM prompt rules — explicit and conservative
LLM_RULES = (
    "You are an extractor that reads the provided SENTENCE(S) and returns percentages that "
    "represent *actual reported reductions* for the TARGET. STRICTLY obey the following:\n"
    "1) TARGET will be either 'HbA1c' or 'Body weight'. Extract only percentages that describe "
    "a reduction/change for that TARGET (not thresholds, not eligibility or composite criteria).\n"
    "2) Prefer explicit within-group change values if that is what the sentence reports for the drug.\n"
    "   If sentence has 'from X% to Y%' treat the reported change as (X - Y) percentage points.\n"
    "3) If multiple timepoints appear (e.g., '6 months'/'T6' and '12 months'/'T12'), prefer 12-month (T12) values.\n"
    "4) If multiple drugs appear, look at the provided 'drug_name' context and prefer the value for that drug.\n"
    "5) Return JSON ONLY with keys: {\"extracted\": [\"...\",\"...\"], \"selected_percent\": \"...\"}.\n"
    "   Each value must include a '%' sign and be numeric (e.g. '1.23%').\n"
)

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name: str = None):
    """
    Returns (list_of_extracted_strings, selected_percent_string_or_empty)
    If model is None or sentence empty → returns ([], '')
    """
    if model is None or not sentence or not sentence.strip():
        return [], ""

    sentence = normalize_text(sentence)
    context_lines = [f"TARGET: {target_label}"]
    if drug_name:
        context_lines.append(f"DRUG_NAME: {drug_name}")
    context_lines.append("SENTENCES:")
    context_lines.append(sentence)
    context_lines.append("\n\nRULES:\n" + LLM_RULES)

    prompt = "\n".join(context_lines) + "\n\nReturn JSON now."

    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = [_norm_percent(x) for x in (data.get("extracted") or []) if isinstance(x, str)]
            selected = _norm_percent(data.get("selected_percent", "") or "")
            # selected must be empty if no extracted
            if not extracted:
                selected = ""
            # convert negative to positive as requested
            if selected:
                try:
                    num = float(selected.replace("%", "").replace(",", "."))
                    selected = f"{abs(num):.3f}".rstrip('0').rstrip('.') + '%'
                except Exception:
                    # leave as is if parse fails
                    pass
            return extracted, selected
    except Exception:
        pass

    # fallback: nothing (we enforce that selected % must come from LLM)
    return [], ""

# -------------------- Scoring helpers --------------------
def a1c_score_from_percent(selected_percent_str: str):
    if not selected_percent_str:
        return ""
    try:
        v = abs(float(selected_percent_str.replace("%", "").replace(",", ".")))
    except Exception:
        return ""
    # thresholds you gave
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

def weight_score_from_percent(selected_percent_str: str):
    if not selected_percent_str:
        return ""
    try:
        v = abs(float(selected_percent_str.replace("%", "").replace(",", ".")))
    except Exception:
        return ""
    if v >= 22:
        return 5
    if 16 <= v < 22:
        return 4
    if 10 <= v < 16:
        return 3
    if 5 <= v < 10:
        return 2
    if v < 5:
        return 1
    return ""

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (leave blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)
use_llm = st.sidebar.checkbox('Enable Gemini 2.0 Flash LLM extraction (requires key)', value=True)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.csv with an "abstract" column.')
    st.stop()

# ---------- read file robustly ----------
try:
    if uploaded.name.lower().endswith('.csv'):
        # try utf-8 first, fallback to latin1
        try:
            df = pd.read_csv(uploaded, encoding='utf-8', dtype=str, keep_default_na=False)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding='latin1', dtype=str, keep_default_na=False)
    else:
        # for excel: if sheet_name blank -> first sheet (0)
        sheet = sheet_name if sheet_name else 0
        df = pd.read_excel(uploaded, sheet_name=sheet, dtype=str)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

# ensure df is DataFrame (not dict)
if isinstance(df, dict):
    # pick first sheet
    first_key = list(df.keys())[0]
    df = df[first_key]

# normalize columns to strings
df = df.fillna("")
st.success(f'Loaded {len(df)} rows and columns: {list(df.columns)}')

# -------------------- Process rows (no st.cache to avoid model hashing issues) --------------------
model = configure_gemini(API_KEY) if (use_llm and API_KEY and GENAI_AVAILABLE) else None
if use_llm and not GENAI_AVAILABLE:
    st.warning("google.generativeai library not available — LLM disabled. Install 'google-generativeai' and restart.")
if use_llm and API_KEY == "<REPLACE_WITH_YOUR_KEY>":
    st.warning("You have not replaced the placeholder API_KEY — LLM will be disabled until you insert your real key.")

rows = []
for idx, row in df.iterrows():
    # original row fields preserved
    new = row.to_dict()

    # normalize and sanitize text
    raw_text = normalize_text(str(row.get(col_name, "") or ""))
    # create sentence columns using regex logic (these are what LLM will read)
    hba_matches, hba_sentences = extract_sentences(raw_text, re_hba1c, 'hba1c')
    wt_matches, wt_sentences = extract_sentences(raw_text, re_weight, 'weight')

    # format regex-extracted values
    def fmt_extracted_vals(matches):
        out = []
        for m in matches:
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                out.append(fmt_pct_val(m.get('reduction_pp')))
            else:
                raw = m.get('raw', '') or ''
                # clean raw percent token if possible
                p = re.search(PCT, raw)
                if p:
                    out.append(fmt_pct_val(parse_number(p.group(1))))
                else:
                    out.append(raw)
        return out

    hba_regex_vals = fmt_extracted_vals(hba_matches)
    wt_regex_vals = fmt_extracted_vals(wt_matches)

    # join sentences (keep in same column); include both important sentences if multiple
    sentence_col = ' | '.join(hba_sentences) if hba_sentences else ''
    weight_sentence_col = ' | '.join(wt_sentences) if wt_sentences else ''

    new['sentence'] = sentence_col
    new['extracted_matches'] = hba_regex_vals
    new['reductions_pp'] = [m.get('reduction_pp') for m in hba_matches]
    new['reduction_types'] = [m.get('type') for m in hba_matches]

    new['weight_sentence'] = weight_sentence_col
    new['weight_extracted_matches'] = wt_regex_vals
    new['weight_reductions_pp'] = [m.get('reduction_pp') for m in wt_matches]
    new['weight_reduction_types'] = [m.get('type') for m in wt_matches]

    # LLM extraction: must read the sentence column, and prefer drug_name if present
    drug_name = None
    if 'drug_name' in df.columns:
        drug_name = str(row.get('drug_name') or "").strip()

    # HbA1c LLM (reads sentence_col)
    if model and sentence_col:
        llm_extracted_hba, selected_hba = llm_extract_from_sentence(model, 'HbA1c', sentence_col, drug_name)
    else:
        llm_extracted_hba, selected_hba = [], ""

    # Weight LLM (reads weight_sentence_col)
    if model and weight_sentence_col:
        llm_extracted_wt, selected_wt = llm_extract_from_sentence(model, 'Body weight', weight_sentence_col, drug_name)
    else:
        llm_extracted_wt, selected_wt = [], ""

    # If LLM gave nothing, enforce selected empty (do not fallback to regex-selected)
    if not llm_extracted_hba:
        selected_hba = ""
    if not llm_extracted_wt:
        selected_wt = ""

    # Put LLM outputs next to regex columns as requested
    new['LLM extracted'] = llm_extracted_hba
    new['selected %'] = selected_hba

    new['Weight LLM extracted'] = llm_extracted_wt
    new['Weight selected %'] = selected_wt

    # compute scores (A1c Score and Weight Score) based on selected%
    new['A1c Score'] = a1c_score_from_percent(selected_hba) if selected_hba else ""
    new['Weight Score'] = weight_score_from_percent(selected_wt) if selected_wt else ""

    rows.append(new)

out_df = pd.DataFrame(rows)

# Keep rows rule: keep if HbA1c OR Weight qualifies (same as earlier)
def _list_has_items(x):
    return isinstance(x, list) and len(x) > 0

mask_hba = (out_df['sentence'].astype(str).str.len() > 0) & (out_df['extracted_matches'].apply(_list_has_items))
mask_wt  = (out_df['weight_sentence'].astype(str).str.len() > 0) & (out_df['weight_extracted_matches'].apply(_list_has_items))
mask_keep = mask_hba | mask_wt
out_df = out_df[mask_keep].reset_index(drop=True)

# counts
out_df.attrs['counts'] = dict(
    kept=int(mask_keep.sum()),
    total=int(len(mask_keep)),
    hba_only=int((mask_hba & ~mask_wt).sum()),
    wt_only=int((mask_wt & ~mask_hba).sum()),
    both=int((mask_hba & mask_wt).sum()),
)

# -------------------- Reorder columns: place LLM columns beside regex columns --------------------
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

# Optionally hide debug
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

st.write("### Results (first 200 rows shown)")
st.dataframe(display_df.head(200))

# show counts
counts = out_df.attrs.get('counts', None)
if counts:
    kept, total = counts['kept'], counts['total']
    st.caption(
        f"Kept {kept} of {total} rows ({(kept/total if total else 0):.1%}).  "
        f"HbA1c-only: {counts['hba_only']}, Weight-only: {counts['wt_only']}, Both: {counts['both']}"
    )

# -------------------- Download results --------------------
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
    file_name='results_with_llm_from_sentence.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
