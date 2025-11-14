# streamlit_hba1c_weight_llm.py
import re
import math
import json
import unicodedata
from io import BytesIO

import pandas as pd
import streamlit as st

# -------------------------- HARD-CODED GEMINI KEY --------------------------
# Replace with your actual key if you want LLM-enabled extraction.
# If you don't want to hardcode a key, set API_KEY = "" and disable the "Use LLM" toggle.
API_KEY = "PASTE_YOUR_GEMINI_KEY_HERE"
# ---------------------------------------------------------------------------

# Lazy import so the app still runs without the package
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (sentence-driven LLM)")

# -------------------------- Regex helpers (original logic) --------------------------
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

# -------------------------- Utilities --------------------------
def parse_number(s: str) -> float:
    if s is None:
        return float('nan')
    s = str(s).replace(',', '.').replace('·', '.').strip()
    try:
        return float(s)
    except Exception:
        return float('nan')

def normalize_mojibake(text: str) -> str:
    """Fix common mojibake/unicode issues and normalize."""
    if not isinstance(text, str):
        return text
    # Some common fixes
    fixed = text.replace('\ufeff', '') \
                .replace('Â·', '·') \
                .replace('Â', '') \
                .replace('â‰¥', '≥') \
                .replace('\x96', '-') \
                .replace('\x97', '-') \
                .replace('\x92', "'")
    # NFKC normalization helps in some cases
    fixed = unicodedata.normalize('NFKC', fixed)
    return fixed

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

# -------------------------- Core extraction (unchanged logic) --------------------------
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

        # reduced/decreased/... by X%
        for m in re_reduce_by.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:percent_or_pp_pmSpaces5', [v], v)

        # reduction of X%
        for m in re_abs_pp.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:pp_word_pmSpaces5', [v], v)

        # ranges like 1.0–1.5% (represent as max)
        for m in re_range.finditer(seg):
            any_window_hit = True
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            rep = None if (math.isnan(a) or math.isnan(b)) else max(a, b)
            add_match(matches, si, abs_s, m, f'{tag_prefix}:range_percent_pmSpaces5', [a, b], rep)

        # any stray percent in the window
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

# -------------------------- Gemini helpers --------------------------
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

# Prompt rules:
LLM_RULES = (
    "You are an information extraction assistant.\n"
    "Read the SENTENCE(S) provided and extract ONLY percentages that correspond to the TARGET (HbA1c or Body weight).\n"
    "- If the sentence contains both within-group value(s) and a between-group difference, prefer the WITHIN-GROUP value(s) for the drug mentioned in 'drug_name' (if provided).\n"
    "- If multiple timepoints exist, prefer 12 months / T12 over 6 months / T6 (keywords: '12 months', 'T12', '12-mo', '6 months', 'T6', '6-mo').\n"
    "- If 'from X% to Y%' appears, compute the change as (X - Y) percentage points and include it as a candidate.\n"
    "- Output STRICT JSON only with keys: {\"extracted\": [\"1.23%\",\"0.8%\"], \"selected_percent\":\"1.23%\"}.\n"
    "- Do not include unrelated percentages (e.g., thresholds like '≥5%' or proportions)."
)

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name: str = None):
    """Call Gemini to extract target percentages from the provided sentence text.
       Returns (list_of_extracted_strings, selected_string_or_empty)."""
    if model is None or not sentence or sentence.strip() == "":
        return [], ""
    # Build prompt
    drug_hint = f"Prefer values for the drug named: {drug_name}." if drug_name else ""
    prompt = (
        f"TARGET: {target_label}\n\n"
        f"{LLM_RULES}\n\n"
        f"{drug_hint}\n\n"
        f"SENTENCE(S):\n{sentence}\n\n"
        "Return JSON only."
    )
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = [_norm_percent(x) for x in (data.get("extracted") or []) if isinstance(x, str)]
            selected = _norm_percent(data.get("selected_percent", "") or "")
            return extracted, selected
    except Exception:
        # safe fallback: return empty (do not fallback to regex)
        return [], ""
    return [], ""

# -------------------------- Scoring utilities --------------------------
def compute_a1c_score(selected_pct_str: str):
    """Return score 1-5 based on absolute selected percent (selected_pct_str like '1.23%')."""
    if not selected_pct_str:
        return ""
    try:
        num = float(selected_pct_str.replace('%', '').replace(',', '.'))
        num = abs(num)
    except Exception:
        return ""
    # mapping per your bands (A1c)
    if num > 2.2:
        return 5
    if 1.8 <= num <= 2.2:
        return 4
    if 1.2 <= num <= 1.7 or (1.2 <= num < 1.8):  # interpreting your ranges (fixing typos)
        return 3
    if 0.8 <= num <= 1.1:
        return 2
    if num < 0.8:
        return 1
    return ""

def compute_weight_score(selected_pct_str: str):
    if not selected_pct_str:
        return ""
    try:
        num = float(selected_pct_str.replace('%', '').replace(',', '.'))
        num = abs(num)
    except Exception:
        return ""
    if num >= 22:
        return 5
    if 16 <= num < 22:
        return 4
    if 10 <= num < 16:
        return 3
    if 5 <= num < 10:
        return 2
    if num < 5:
        return 1
    return ""

# -------------------------- UI --------------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
# If file uploaded, we will let user select which column contains abstracts
text_col_selected = None
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
use_llm = st.sidebar.checkbox('Enable Gemini LLM (Gemini 2.0 Flash)', value=bool(API_KEY))
# Choose drug_name column (optional)
drug_col_name = st.sidebar.text_input('Drug name column (optional, leave blank if none)', value='drug_name')

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar.')
    st.stop()

# read file robustly
try:
    if uploaded.name.lower().endswith('.csv'):
        # Try to read with utf-8; if fails, fall back to latin-1 and re-encode
        try:
            df = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            df = pd.read_csv(uploaded, encoding='latin1')
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else 0)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

# Allow user to pick the text column from available columns
st.sidebar.markdown("### Choose columns")
col_name = st.sidebar.selectbox('Column with abstracts/text', options=list(df.columns), index=0)
# optional weight/drug columns could be chosen too (we keep drug as text input for flexibility)
if drug_col_name and drug_col_name not in df.columns:
    # if they typed a name that doesn't exist, show a small note but allow blank to mean none
    st.sidebar.warning(f"Drug column '{drug_col_name}' not found in uploaded file. Leave blank or type existing column name.")
# show debug toggle
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

st.success(f'Loaded {len(df)} rows and column choices. Processing...')

# configure gemini if requested
model = None
if use_llm:
    model = configure_gemini(API_KEY)
    if model is None:
        st.sidebar.error("Gemini model not configured or API key missing/invalid. LLM columns will be empty.")
        use_llm = False  # fallback

# -------------------------- PROCESS (NO caching to avoid UnhashableParamError) --------------------------
rows = []
for _, row in df.iterrows():
    # read and normalize text
    text_raw = row.get(col_name, '') if col_name in row else row.iloc[0] if len(row)>0 else ''
    if not isinstance(text_raw, str):
        text_raw = str(text_raw)
    text = normalize_mojibake(text_raw)

    # Extract regex-based sentences and matches
    hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')
    # STRICT FILTER for HbA1c: keep only values < 7 (prefer reduction_pp if present)
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

    # Prepare sentence strings: join multiple qualifying sentences with " | "
    sentence_str = ' | '.join(hba_sentences) if hba_sentences else ''
    weight_sentence_str = ' | '.join(wt_sentences) if wt_sentences else ''

    # Regex-extracted values (strings)
    hba_regex_vals = [fmt_extracted_hba(m) for m in hba_matches]
    wt_regex_vals = [fmt_extracted_wt(m) for m in wt_matches]

    # LLM step: reads the sentence strings (not the whole abstract)
    hba_llm_extracted, hba_selected = [], ""
    wt_llm_extracted, wt_selected = [], ""

    # drug hint
    drug_hint_val = ""
    if drug_col_name and drug_col_name in df.columns:
        drug_hint_val = str(row.get(drug_col_name, "") or "").strip()

    if use_llm and model is not None:
        # HbA1c LLM
        if sentence_str:
            ex, sel = llm_extract_from_sentence(model, "HbA1c", sentence_str, drug_hint_val)
            # selected must be empty if ex empty
            if ex:
                hba_llm_extracted, hba_selected = ex, sel
            else:
                hba_llm_extracted, hba_selected = [], ""
        # Weight LLM
        if weight_sentence_str:
            exw, selw = llm_extract_from_sentence(model, "Body weight", weight_sentence_str, drug_hint_val)
            if exw:
                wt_llm_extracted, wt_selected = exw, selw
            else:
                wt_llm_extracted, wt_selected = [], ""

    # Normalize selected %: make positive, ensure % suffix; if empty keep empty
    def make_positive_pct(s):
        if not s:
            return ""
        s2 = s.replace('%', '').replace(',', '.').strip()
        try:
            val = abs(float(s2))
            # remove trailing zeros appropriately
            s_form = (f"{val:.3f}".rstrip('0').rstrip('.')) + '%'
            return s_form
        except Exception:
            return ""

    hba_selected = make_positive_pct(hba_selected)
    wt_selected  = make_positive_pct(wt_selected)

    # Scores based on selected %
    a1c_score = compute_a1c_score(hba_selected)
    weight_score = compute_weight_score(wt_selected)

    # Build output row: preserve all original columns
    new = row.to_dict()
    new.update({
        'sentence': sentence_str,
        'extracted_matches': hba_regex_vals,
        'LLM extracted': hba_llm_extracted,
        'selected %': hba_selected,
        'A1c Score': a1c_score,
        'weight_sentence': weight_sentence_str,
        'weight_extracted_matches': wt_regex_vals,
        'Weight LLM extracted': wt_llm_extracted,
        'Weight selected %': wt_selected,
        'Weight Score': weight_score,
    })
    rows.append(new)

out = pd.DataFrame(rows)

# Keep rows where either HbA1c or weight had regex matches (same rule as before)
def _list_has_items(x):
    return isinstance(x, list) and len(x) > 0

mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(_list_has_items))
mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(_list_has_items))
mask_keep = mask_hba | mask_wt
out = out[mask_keep].reset_index(drop=True)

# Ensure selected% empty if LLM extracted was empty (explicitly)
out['selected %'] = out.apply(lambda r: r['selected %'] if isinstance(r['LLM extracted'], list) and len(r['LLM extracted'])>0 else "", axis=1)
out['Weight selected %'] = out.apply(lambda r: r['Weight selected %'] if isinstance(r['Weight LLM extracted'], list) and len(r['Weight LLM extracted'])>0 else "", axis=1)

# Reorder columns to place LLM columns beside regex ones
def insert_after(cols, after, names):
    if after not in cols:
        return cols
    i = cols.index(after)
    for name in names[::-1]:
        if name in cols:
            cols.remove(name)
        cols.insert(i+1, name)
    return cols

display_df = out.copy()
cols = list(display_df.columns)
cols = insert_after(cols, "extracted_matches", ["LLM extracted", "selected %", "A1c Score"])
cols = insert_after(cols, "weight_extracted_matches", ["Weight LLM extracted", "Weight selected %", "Weight Score"])
display_df = display_df[cols]

# Clean output values for mojibake before displaying/exporting
for c in display_df.select_dtypes(include=[object]).columns:
    display_df[c] = display_df[c].apply(lambda x: normalize_mojibake(x) if isinstance(x, str) else x)

# UI display
st.write('### Results (first 200 rows shown)')
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

st.dataframe(display_df.head(200))

# counts (if available earlier)
# (We didn't track counts the exact same way here; show simple counts)
st.caption(f"Rows kept: {len(display_df)} / {len(df)}")

# Download cleaned Excel
@st.cache_data
def to_excel_bytes(df_in):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_in.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(display_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_llm_and_scores.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

st.markdown('---')
st.write("Notes & behaviors:")
st.write("- LLM reads the **sentence** (or weight_sentence) column and extracts target percentages only.")
st.write("- If LLM extracted list is empty, `selected %` remains empty (no fallback to regex).")
st.write("- The script normalizes common mojibake artifacts (e.g., replaces 'Â·' etc.).")
st.write("- If multiple drugs appear in sentence and `drug_name` column is supplied & matches, LLM is asked to prefer that drug's values.")
st.write("- Scores are absolute (positive) and placed next to the selected % columns.")

