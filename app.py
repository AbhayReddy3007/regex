# streamlit_hba1c_weight_llm.py
import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
# Paste your Gemini key between the quotes below.
API_KEY = ""   # <- Paste your Gemini API key here (e.g. "sk-..."). Keep it secret!
# =========================================================================

# Lazy Gemini import so the app still runs without the package
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (reads sentence column)")

# -------------------- Regex helpers (robust) --------------------
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

# Terms & cues
re_hba1c  = re.compile(r'\bhb\s*a1c\b|\bhba1c\b|\ba1c\b', FLAGS)
re_weight = re.compile(r'\b(body\s*weight|weight|bw)\b', FLAGS)
re_reduction_cue = re.compile(
    r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing)|decrease|decreased|reduction|reduced|improved)\b',
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
    """Conservative sentence splitter on ., ?, ! or newlines."""
    if not isinstance(text, str):
        return []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # split on sentence-end punctuation or on two+ newlines; keep sentences fairly long
    parts = re.split(r'(?<=[\.!\?])\s+|\n{2,}|\n', text)
    return [p.strip() for p in parts if p and p.strip()]

def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    """Require: (target term) AND a % AND a reduction cue OR explicit 'mean decrease' style."""
    # Must contain target (HbA1c or weight) and a percent number
    if not term_re.search(sent):
        return False
    if not re_pct.search(sent):
        return False
    # If it has an explicit reduction cue or words like 'mean decrease' or 'yielded'
    if re_reduction_cue.search(sent):
        return True
    if re.search(r'\bmean\s+(?:decrease|decreases|decreased|change)\b', sent, re.IGNORECASE):
        return True
    if re.search(r'\byielded\b', sent, re.IGNORECASE):
        return True
    # otherwise prefer to skip (avoid thresholds/proportions)
    return False

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

# -------------------- Core extraction --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    """
    Within a qualifying sentence:
      • Scan the WHOLE SENTENCE for 'from X% to Y%' and record deltas.
      • For all other % patterns, search ONLY within a ±5-SPACES window
        around each term occurrence.
      • Weight: nearest previous % within 60 chars if no window hit.
    """
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

        # any percent in the window
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
                # map to sentence-abs indices
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
        # be permissive: accept sentence if it has term and % and a reduction cue or yielding/mean decrease
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
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v

# Build LLM instructions carefully (ignore thresholds / proportions)
LLM_RULES = (
    "You are an extractor for clinical % CHANGES. Read the SENTENCE(S) supplied and do ONLY this:\n"
    " - Extract percentage CHANGE values that represent an outcome change (e.g., 'reduced by 1.2%', 'mean decrease of 1.75%', 'yielded 13.88% weight reduction').\n"
    " - DO NOT extract THRESHOLDS or CRITERIA (phrases like 'achieving ≥5%' or 'HbA1c < 7.0%') — those are outcome thresholds, not change magnitudes.\n"
    " - If the sentence has 'from X% to Y%' or 'declined from X% to Y%', compute the difference (X - Y) and include that in extracted as a percentage string (e.g. '1.3%').\n"
    " - Prefer within-group reported changes (phrases like 'mean decrease', 'yielded X% in [drug group]') over between-group differences (phrases like 'between-group difference:', 'difference: -13.46%').\n"
    " - If multiple timepoints exist (e.g., 'T6','T12','6 months','12 months'), prefer values for 12 months/T12 over 6 months/T6.\n"
    " - If provided, use the DRUG_NAME to prefer percentages that refer to that drug's group.\n"
    " - Return STRICT JSON only, no commentary, with keys:\n"
    "   {\"extracted\": [\"1.75%\",\"0.4%\"], \"selected_percent\": \"1.75%\"}\n"
    " - If you cannot find a change percentage for the target, return {\"extracted\": [], \"selected_percent\": \"\"}.\n"
)

def llm_extract_from_sentence(model_obj, target_label: str, sentence: str, drug_name: str = None):
    """
    Calls Gemini and expects strict JSON back with extracted list and selected_percent.
    Returns (extracted_list_of_strings, selected_percent_string_or_empty).
    """
    if model_obj is None or not sentence or not sentence.strip():
        return [], ""

    prompt = f"TARGET: {target_label}\n"
    if drug_name:
        prompt += f"DRUG_NAME: {drug_name}\n"
    prompt += LLM_RULES + "\nSENTENCE(S):\n" + sentence + "\n\nReturn JSON."

    try:
        resp = model_obj.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = [_norm_percent(x) for x in (data.get("extracted") or []) if isinstance(x, str)]
            selected = _norm_percent(data.get("selected_percent", "")) if data.get("selected_percent") else ""
            # ensure selected is empty if extracted empty
            if not extracted:
                selected = ""
            # convert negative to positive in selected
            if selected:
                try:
                    n = parse_number(selected.replace("%", ""))
                    if not math.isnan(n):
                        selected = _norm_percent(str(abs(n)))
                except Exception:
                    pass
            return extracted, selected
    except Exception:
        # fail gracefully
        pass
    return [], ""

# -------------------- Scoring logic --------------------
def a1c_score_from_pct(pct_str: str):
    """Return score 1-5 given selected% string like '1.23%' (absolute positive)"""
    if not pct_str:
        return ""
    try:
        n = abs(parse_number(pct_str.replace("%", "")))
        if n > 2.2:
            return 5
        if 1.8 <= n <= 2.1:
            return 4
        if 1.2 <= n <= 1.7:
            return 3
        if 0.8 <= n <= 1.1:
            return 2
        if n < 0.8:
            return 1
    except Exception:
        return ""
    return ""

def weight_score_from_pct(pct_str: str):
    if not pct_str:
        return ""
    try:
        n = abs(parse_number(pct_str.replace("%", "")))
        if n >= 22:
            return 5
        if 16 <= n <= 21.9:
            return 4
        if 10 <= n <= 15.9:
            return 3
        if 5 <= n <= 9.9:
            return 2
        if n < 5:
            return 1
    except Exception:
        return ""
    return ""

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
use_llm    = st.sidebar.checkbox('Enable Gemini LLM extraction (Gemini 2.0 Flash)', value=True)
remove_null_rows = st.sidebar.checkbox('Remove rows with no extracted matches', value=True)

if not uploaded:
    st.info('Upload your Excel/CSV file in the left sidebar. The file may have multiple columns (drug_name is used if present).')
    st.stop()

# read file robustly (handle CSVs with weird encoding)
try:
    if uploaded.name.lower().endswith('.csv'):
        # try utf-8 first; fallback to latin1 with replacement
        try:
            df = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            try:
                uploaded.seek(0)
            except Exception:
                pass
            df = pd.read_csv(uploaded, encoding='latin1', errors='replace')
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else 0)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

# Ensure we treat text cells as strings
df[col_name] = df[col_name].fillna('').astype(str)

# -------------------- Processing (cache-friendly: do not hash model) --------------------
@st.cache_data
def process_df(_model, df_in, text_col, use_llm_flag):
    rows = []
    # iterate rows
    for _, row in df_in.iterrows():
        text = row.get(text_col, '') or ''
        # Extract regex-based sentences + matches
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')
        wt_matches, wt_sentences   = extract_sentences(text, re_weight, 'weight')

        # Filter HbA1c by numeric rule: <7 or from-to
        def allowed_hba(m):
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                try:
                    return float(abs(rp)) < 7.0
                except:
                    return False
            for v in (m.get('values') or []):
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and abs(float(v)) < 7.0:
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

        def fmt_extracted_wt(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        hba_regex_vals = [fmt_extracted_hba(m) for m in hba_matches]
        wt_regex_vals  = [fmt_extracted_wt(m) for m in wt_matches]

        # Build sentence strings for LLM to read (joined)
        hba_sentence = " | ".join(hba_sentences) if hba_sentences else ""
        wt_sentence  = " | ".join(wt_sentences) if wt_sentences else ""

        # Drug name for context (if present)
        drug_name = None
        if 'drug_name' in df_in.columns:
            dn = row.get('drug_name', None)
            if isinstance(dn, str) and dn.strip():
                drug_name = dn.strip()

        # LLM extraction (if enabled)
        hba_llm_vals, hba_selected = ([], "")
        wt_llm_vals, wt_selected   = ([], "")

        if use_llm_flag and _model is not None:
            # HbA1c LLM: read the sentence string (if any)
            if hba_sentence:
                try:
                    vals, sel = llm_extract_from_sentence(_model, "HbA1c", hba_sentence, drug_name)
                    # ensure selected is empty if vals empty (we already do this inside)
                    hba_llm_vals, hba_selected = vals, sel
                except Exception:
                    hba_llm_vals, hba_selected = [], ""
            # Weight LLM
            if wt_sentence:
                try:
                    vals, sel = llm_extract_from_sentence(_model, "Body weight", wt_sentence, drug_name)
                    wt_llm_vals, wt_selected = vals, sel
                except Exception:
                    wt_llm_vals, wt_selected = [], ""

        # selected% must be empty if LLM extracted is empty (explicit)
        if not hba_llm_vals:
            hba_selected = ""
        if not wt_llm_vals:
            wt_selected = ""

        # Coerce selected% to positive numeric strings
        def normalize_sel(s):
            if not s:
                return ""
            try:
                n = parse_number(s.replace("%", ""))
                if math.isnan(n):
                    return ""
                return fmt_pct(abs(n))
            except:
                return s
        hba_selected = normalize_sel(hba_selected)
        wt_selected  = normalize_sel(wt_selected)

        # Scores
        a1c_score = a1c_score_from_pct(hba_selected)
        weight_score = weight_score_from_pct(wt_selected)

        new = row.to_dict()
        # Add/overwrite columns
        new.update({
            'sentence': hba_sentence,
            'extracted_matches': hba_regex_vals,
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],

            'LLM extracted': hba_llm_vals,
            'selected %': hba_selected,
            'A1c Score': a1c_score,

            'weight_sentence': wt_sentence,
            'weight_extracted_matches': wt_regex_vals,
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],

            'Weight LLM extracted': wt_llm_vals,
            'Weight selected %': wt_selected,
            'Weight Score': weight_score,
        })

        rows.append(new)

    out = pd.DataFrame(rows)

    # Optionally remove rows with no extracted matches on both targets
    if remove_null_rows:
        def _has_items(x):
            return isinstance(x, list) and len(x) > 0
        mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(_has_items))
        mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(_has_items))
        mask_keep = mask_hba | mask_wt
        out = out[mask_keep].reset_index(drop=True)

    # counts
    out.attrs['counts'] = dict(
        kept=int(len(out)),
        total=int(len(df_in)),
    )
    return out

# run processing (pass model as _model — avoids Streamlit hashing it)
out_df = process_df(model, df, col_name, use_llm)

# Reorder columns to place LLM columns beside regex columns
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
# ensure new order exists
display_df = display_df[[c for c in cols if c in display_df.columns]]

# Show the results
st.write("### Results (first 200 rows shown)")
st.dataframe(display_df.head(200))

# info
counts = out_df.attrs.get('counts', None)
if counts:
    st.caption(f"Kept {counts['kept']} rows (input total {counts['total']}).")

# Download
@st.cache_data
def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(display_df)
st.download_button('Download results as Excel', data=excel_bytes,
                   file_name='results_with_llm_from_sentence.xlsx',
                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.markdown('---')
st.write("**Notes / behaviours implemented:**")
st.write("- LLM reads the `sentence` / `weight_sentence` column(s) and extracts change percentages; `selected %` is the single percent the LLM chooses as best.")
st.write("- If LLM extracted list is empty, `selected %` is forced to empty (no fallback).")
st.write("- Negative selected% values are converted to positive (absolute).")
st.write("- 'from X% to Y%' is converted into a delta (X - Y) and included among extracted values.")
st.write("- LLM instructed to ignore thresholds (e.g., '≥5%' as a target threshold) and to prefer within-group change values where possible.")
st.write("- If your input has `drug_name` column, it is passed to the LLM to prefer values referring to that drug.")
