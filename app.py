# streamlit_hba1c_weight_llm.py
import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# -------------------- HARD-CODE GEMINI KEY (replace below) --------------------
API_KEY = "YOUR_GEMINI_API_KEY_HERE"   # <--- REPLACE with your key (local only)
# -----------------------------------------------------------------------------

# lazy import so app still runs without package
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini (reads sentence columns)")

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

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

def norm_percent_str(v: str) -> str:
    """Normalize string like ' -13.46 %' -> '13.46%' (but keep sign info separately)"""
    if not v:
        return ''
    s = str(v).strip().replace(' ', '').replace('·', '.')
    if s.endswith('%'):
        s = s[:-1]
    # remove parentheses or commas
    s = re.sub(r'[^\d\+\-\.]', '', s)
    if s == '':
        return ''
    # ensure numeric
    if re.match(r'^[+-]?\d+(\.\d+)?$', s):
        # return with % sign
        return (s.lstrip('+')) + '%'
    return ''

# --- window by spaces function (used by original logic) ---
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

def add_match(out, si, abs_start, m, typ, values, reduction):
    out.append({
        'raw': m.group(0),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (abs_start + m.start(), abs_start + m.end()),
    })

# -------------------- Core extraction (cached, no model) --------------------
@st.cache_data
def process_df_regex_only(df: pd.DataFrame, text_col: str, drug_col: str|None = None):
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        text = '' if not isinstance(text, str) else text

        # HbA1c extraction
        hba_matches, hba_sentences = [], []
        for si, sent in enumerate(split_sentences(text)):
            if not (re_hba1c.search(sent) and re_pct.search(sent) and re_reduction_cue.search(sent)):
                continue
            hba_sentences.append(sent)
            # from-to full sentence
            for m in re_fromto.finditer(sent):
                a = parse_number(m.group(1)); b = parse_number(m.group(2))
                red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
                add_match(hba_matches, si, 0, m, 'hba1c:from-to_sentence', [a,b], red)
            # windows
            for hh in re_hba1c.finditer(sent):
                seg, abs_s, _ = window_prev_next_spaces_inclusive_tokens(sent, hh.end(), 5, 5)
                for m in re_reduce_by.finditer(seg):
                    v = parse_number(m.group(1)); add_match(hba_matches, si, abs_s, m, 'hba1c:percent_or_pp_pmSpaces5', [v], v)
                for m in re_abs_pp.finditer(seg):
                    v = parse_number(m.group(1)); add_match(hba_matches, si, abs_s, m, 'hba1c:pp_word_pmSpaces5', [v], v)
                for m in re_range.finditer(seg):
                    a = parse_number(m.group(1)); b = parse_number(m.group(2)); rep = None if (math.isnan(a) or math.isnan(b)) else max(a,b)
                    add_match(hba_matches, si, abs_s, m, 'hba1c:range_percent_pmSpaces5', [a,b], rep)
                for m in re_pct.finditer(seg):
                    v = parse_number(m.group(1)); add_match(hba_matches, si, abs_s, m, 'hba1c:percent_pmSpaces5', [v], v)

        # dedupe and keep only allowed (<7 rule)
        def allowed_hba(m):
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                return float(abs(rp)) < 7.0  # allow sign but compare absolute
            for v in (m.get('values') or []):
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and float(abs(v)) < 7.0:
                        return True
                except Exception:
                    pass
            return False

        hba_matches = [m for m in hba_matches if allowed_hba(m)]
        hba_matches.sort(key=lambda x:(x['sentence_index'], x['span'][0]))

        # Weight extraction
        wt_matches, wt_sentences = [], []
        for si, sent in enumerate(split_sentences(text)):
            if not (re_weight.search(sent) and re_pct.search(sent) and re_reduction_cue.search(sent)):
                continue
            wt_sentences.append(sent)
            # from-to full sentence
            for m in re_fromto.finditer(sent):
                a = parse_number(m.group(1)); b = parse_number(m.group(2))
                red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
                add_match(wt_matches, si, 0, m, 'weight:from-to_sentence', [a,b], red)
            # windows
            for hh in re_weight.finditer(sent):
                seg, abs_s, _ = window_prev_next_spaces_inclusive_tokens(sent, hh.end(), 5, 5)
                for m in re_reduce_by.finditer(seg):
                    v = parse_number(m.group(1)); add_match(wt_matches, si, abs_s, m, 'weight:percent_or_pp_pmSpaces5', [v], v)
                for m in re_abs_pp.finditer(seg):
                    v = parse_number(m.group(1)); add_match(wt_matches, si, abs_s, m, 'weight:pp_word_pmSpaces5', [v], v)
                for m in re_range.finditer(seg):
                    a = parse_number(m.group(1)); b = parse_number(m.group(2)); rep = None if (math.isnan(a) or math.isnan(b)) else max(a,b)
                    add_match(wt_matches, si, abs_s, m, 'weight:range_percent_pmSpaces5', [a,b], rep)
                for m in re_pct.finditer(seg):
                    v = parse_number(m.group(1)); add_match(wt_matches, si, abs_s, m, 'weight:percent_pmSpaces5', [v], v)
            # fallback: nearest previous % within 60 chars
            # implemented in original logic: only if windows didn't hit - here simplify: if no matches in this sentence, search left
            if not any(m['sentence_index']==si for m in wt_matches):
                for hh in re_weight.finditer(sent):
                    pos = hh.start()
                    left = max(0, pos - 60)
                    left_chunk = sent[left:pos]
                    last_pct = None
                    for m in re_pct.finditer(left_chunk):
                        last_pct = m
                    if last_pct is not None:
                        abs_start = left
                        v = parse_number(last_pct.group(1))
                        add_match(wt_matches, si, abs_start, last_pct, 'weight:percent_prev60chars', [v], v)

        wt_matches.sort(key=lambda x:(x['sentence_index'], x['span'][0]))

        # Build output row
        new = row.to_dict()
        new.update({
            'sentence': ' | '.join(hba_sentences) if hba_sentences else '',
            'extracted_matches': [ (fmt_pct(m['reduction_pp']) if 'from-to' in m['type'] else m['raw']) for m in hba_matches],
            'reductions_pp': [ m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [ m.get('type') for m in hba_matches],
            'weight_sentence': ' | '.join(wt_sentences) if wt_sentences else '',
            'weight_extracted_matches': [ (fmt_pct(m['reduction_pp']) if 'from-to' in m['type'] else m['raw']) for m in wt_matches],
            'weight_reductions_pp': [ m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [ m.get('type') for m in wt_matches],
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # Keep rows where either HbA1c or weight qualifies
    def _list_has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(_list_has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(_list_has_items))
    mask_keep = mask_hba | mask_wt
    out = out[mask_keep].reset_index(drop=True)

    # counts attr
    out.attrs['counts'] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum()),
    )
    return out

# -------------------- Gemini helper (non-cached usage) --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")  # user requested 2.0 flash
    except Exception:
        return None

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name: str|None = None):
    """
    Ask LLM (Gemini) to extract reduction percentages for TARGET from the provided sentence text.
    Returns (list_of_extracted_strings, selected_percent_string_or_empty)
    Rules:
      - Only extract reduction percentages (ignore threshold statements like 'HbA1c < 7.0%' that are not reductions).
      - Prefer T12/12 months/T12 over T6/6 months/T6.
      - Prefer numbers associated with drug_name (if provided).
      - If between-group difference present (word 'between-group' or 'between group'), prefer that (but LLM decides).
      - Return strict JSON: {"extracted":["1.23%"], "selected_percent":"1.23%"}
    """
    if model is None or not sentence or not sentence.strip():
        return [], ""

    prompt = (
        f"TARGET: {target_label}\n"
        f"DRUG_NAME: {drug_name or ''}\n\n"
        "You are an information-extraction assistant. Read the sentence(s) carefully and:\n"
        "- Extract ONLY percentage values that represent ACTUAL reported REDUCTIONS for the TARGET (HbA1c or body weight).\n"
        "- Do NOT extract threshold/target definitions or proportions (e.g. 'HbA1c < 7.0%' as an endpoint threshold) unless it is explicitly a reported reduction.\n"
        "- If the sentence reports both within-group and between-group numbers, prefer the between-group difference when it explicitly says 'between-group' or 'between group'.\n"
        "- If multiple timepoints are present (e.g. T6, T12, 6 months, 12 months), prefer 12/T12 over 6/T6.\n"
        "- Prefer numbers that are associated with the provided DRUG_NAME (if present) over placebo or comparator numbers.\n"
        "- Normalize numbers to a consistent format with % sign. If a number is negative, keep its sign in extracted list but selected percent will be positive.\n"
        "Return JSON ONLY with keys: {\"extracted\": [\"...\",\"...\"], \"selected_percent\": \"...\"}. Example: {\"extracted\":[\"13.88%\",\"-13.46%\"], \"selected_percent\":\"13.46%\"}\n\n"
        f"SENTENCE: {sentence}\n"
    )
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # extract JSON object
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e > s:
            js = text[s:e+1]
            data = json.loads(js)
            extracted_raw = data.get("extracted") or []
            selected_raw = data.get("selected_percent", "") or ""
            # normalize strings
            extracted = []
            for x in extracted_raw:
                xs = norm_percent_str(x)
                if xs:  # keep sign? we'll keep sign in list by reading numeric sign from original if present
                    # preserve sign if original had '-' explicitly
                    if isinstance(x, str) and x.strip().startswith('-') and not xs.startswith('-'):
                        xs = '-' + xs.lstrip('+').lstrip('-')
                    extracted.append(xs)
            sel = norm_percent_str(selected_raw)
            # selected percent should be positive in output per user request
            if sel.startswith('-'):
                sel = sel.lstrip('-')
            return extracted, sel
    except Exception:
        # fall through to heuristic fallback if LLM fails
        pass

    # fallback heuristics (if LLM fails)
    # extract any percent-looking tokens in sentence, prefer between-group token if present
    extracted = []
    for m in re.findall(r'[+-]?\d+(?:[.,·]\d+)?\s*%', sentence):
        extracted.append(norm_percent_str(m))
    # prefer between-group difference
    bg = None
    for m in re.finditer(r'between[- ]group.*?([+-]?\d+(?:[.,·]\d+)?)\s*%', sentence, flags=re.IGNORECASE):
        bg = norm_percent_str(m.group(1))
        break
    selected = bg or (extracted[0] if extracted else "")
    if selected.startswith('-'):
        selected = selected.lstrip('-')
    return extracted, selected

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
drug_col   = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

if not uploaded:
    st.info('Upload your CSV/XLSX file in the left sidebar (it can have multiple columns including abstract and drug_name).')
    st.stop()

# read file robustly with utf-8 fallback
try:
    if uploaded.name.lower().endswith('.csv'):
        try:
            df = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding='latin1')
    else:
        # excel: try reading
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else 0)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

# run regex-only extraction (cached)
out_df = process_df_regex_only(df, col_name, drug_col if drug_col in df.columns else None)

# configure model (non-cached)
model = configure_gemini(API_KEY) if API_KEY else None
if model is None and API_KEY:
    st.warning("Gemini client not available or API_KEY invalid; LLM columns will be empty.")

# Run LLM extraction on sentence columns (not cached to avoid hashing model)
# HbA1c
hba_llm_results = []
for idx, row in out_df.iterrows():
    sentence_text = str(row.get('sentence', '') or '')
    drug_name_val = str(row.get(drug_col, '') or '') if drug_col in out_df.columns else None
    extracted_list, selected = llm_extract_from_sentence(model, "HbA1c", sentence_text, drug_name_val)
    # selected should be positive
    if selected and selected.startswith('-'):
        selected = selected.lstrip('-')
    hba_llm_results.append((extracted_list, selected or ""))

out_df['LLM extracted'] = [r for r, s in hba_llm_results]
out_df['selected %'] = [s for r, s in hba_llm_results]

# Weight
wt_llm_results = []
for idx, row in out_df.iterrows():
    sentence_text = str(row.get('weight_sentence', '') or '')
    drug_name_val = str(row.get(drug_col, '') or '') if drug_col in out_df.columns else None
    extracted_list, selected = llm_extract_from_sentence(model, "Body weight", sentence_text, drug_name_val)
    if selected and selected.startswith('-'):
        selected = selected.lstrip('-')
    wt_llm_results.append((extracted_list, selected or ""))

out_df['Weight LLM extracted'] = [r for r, s in wt_llm_results]
out_df['Weight selected %'] = [s for r, s in wt_llm_results]

# Convert selected % strings to numeric values for scoring (strip %)
def selected_to_float(s):
    try:
        if not s:
            return float('nan')
        s2 = str(s).strip().replace('%','').replace('·','.')
        return abs(float(s2))
    except Exception:
        return float('nan')

out_df['selected_%_num'] = out_df['selected %'].apply(selected_to_float)
out_df['weight_selected_%_num'] = out_df['Weight selected %'].apply(selected_to_float)

# -------------------- Scoring --------------------
def a1c_score_from_pct(p):
    # Scores for A1c
    # 5: >2.2
    # 4: 1.8 - 2.1
    # 3: 1.2 - 1.7  (note your text had '3.1.2%' - fixed to 1.2-1.7)
    # 2: 0.8 - 1.1
    # 1: <0.8
    if p is None or math.isnan(p):
        return ''
    if p > 2.2:
        return 5
    if 1.8 <= p <= 2.1:
        return 4
    if 1.2 <= p <= 1.7:
        return 3
    if 0.8 <= p <= 1.1:
        return 2
    if p < 0.8:
        return 1
    # catch-alls
    return ''

def weight_score_from_pct(p):
    # 5: >=22%
    # 4: 16 - 21.9
    # 3: 10 - 15.9
    # 2: 5 - 9.9
    # 1: <5
    if p is None or math.isnan(p):
        return ''
    if p >= 22.0:
        return 5
    if 16.0 <= p <= 21.9:
        return 4
    if 10.0 <= p <= 15.9:
        return 3
    if 5.0 <= p <= 9.9:
        return 2
    if p < 5.0:
        return 1
    return ''

out_df['A1c Score'] = out_df['selected_%_num'].apply(a1c_score_from_pct)
out_df['Weight Score'] = out_df['weight_selected_%_num'].apply(weight_score_from_pct)

# drop the numeric helper columns if you want; keep for debug if show_debug true
if not show_debug:
    if 'selected_%_num' in out_df.columns: out_df = out_df.drop(columns=['selected_%_num', 'weight_selected_%_num'])

# -------------------- Place LLM columns beside regex columns --------------------
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

# -------------------- Show results --------------------
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
    with pd.ExcelWriter(buffer) as writer:
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
