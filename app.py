# streamlit_hba1c_weight_llm.py
import re
import math
import json
import unicodedata
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
# (You asked for hard-coded key; replace if you want another)
API_KEY = "WEJKEBHABLRKJVBR;KEARVBBVEKJ"
# ==========================================================================

# Lazy import for Gemini so the app still runs without the package
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (sentence-based)")

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

# -------------------- Utilities & cleaning --------------------
def clean_text_artifacts(s: str) -> str:
    """Normalize unicode and common mojibake sequences so Excel output is clean."""
    if not isinstance(s, str):
        return ""
    # Normalize unicode nicely first
    s = unicodedata.normalize("NFKC", s)
    # Common mojibake replacements seen in PDFs/CSVs:
    s = s.replace("Â·", "·")  # middle dot
    s = s.replace("â‰¥", "≥")
    s = s.replace("â€“", "–")
    s = s.replace("â€”", "—")
    s = s.replace("\u00a0", " ")  # NBSP to space
    # Replace weird control characters
    s = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', s)
    # Trim repeated spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

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

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

def abs_pct_str(s: str) -> str:
    """Return normalized percent string and ensure positive sign (no minus)."""
    if not s:
        return ''
    s = s.strip().replace(" ", "")
    if s.endswith('%'):
        core = s[:-1]
    else:
        core = s
    try:
        val = float(core.replace(',', '.').replace('·', '.'))
        return fmt_pct(abs(val))
    except Exception:
        return s if s.endswith('%') else (s + '%')

# --- local window by spaces (inclusive token)
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

# -------------------- Core regex extraction --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []

    # 1) WHOLE-SENTENCE: from X% to Y% -> delta
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

# -------------------- Gemini 2.0 Flash helpers --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

# LLM instructions: strict about reductions only, ignore threshold statements,
# prefer T12/Twelve-month values over T6, and focus on the target label.
LLM_RULES = (
    "You are an extractor. Read the provided SENTENCE and extract ONLY percentage values that represent "
    "a REPORTED REDUCTION for the TARGET. Do NOT extract thresholds, inclusion criteria, or unrelated percentages. "
    "If the sentence contains 'from X% to Y%', interpret the reported reduction as (X - Y) percentage points and include that as a value. "
    "If multiple timepoints are present, prefer 12-month/T12/12-mo over 6-month/T6 values. "
    "Return STRICT JSON only, e.g. {\"extracted\":[\"1.23%\",\"0.85%\"],\"selected_percent\":\"1.23%\"}. "
    "Do not include any commentary or other keys."
)

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if not v:
        return ""
    if v.endswith('%'):
        core = v[:-1]
    else:
        core = v
    try:
        num = float(core.replace(',', '.').replace('·', '.'))
        return fmt_pct(abs(num))
    except Exception:
        # as fallback, keep original cleaned with percent if missing
        if not v.endswith('%'):
            v = v + '%'
        return v

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name: str = None):
    """
    Ask LLM to read sentence and extract reduction percentages for the target.
    model: configured gemini model or None
    """
    sentence = clean_text_artifacts(sentence)
    if model is None or not sentence.strip():
        return [], ""

    prompt = (
        f"TARGET: {target_label}\n"
        f"{'DRUG:'+drug_name+'\\n' if drug_name else ''}"
        f"{LLM_RULES}\n\nSENTENCE:\n{sentence}\n"
        "Return JSON only.\n"
    )

    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # extract JSON substring
        s = text.find("{"); e = text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = []
            for x in data.get("extracted", []) if isinstance(data.get("extracted", []), list) else []:
                extracted.append(_norm_percent(x))
            selected = _norm_percent(data.get("selected_percent", ""))
            # ensure selected is one of extracted if possible; otherwise accept selected
            if selected and extracted and selected not in extracted:
                # keep selected but ensure normalized format
                pass
            return extracted, selected
    except Exception:
        # silent fallback
        return [], ""
    return [], ""

# -------------------- Score functions --------------------
def a1c_score_from_pct_str(pct_str: str) -> int:
    """Scores:
    5: >2.2%
    4: 1.8%-2.1%
    3: 1.2%-1.7%
    2: 0.8%-1.1%
    1: <0.8%
    """
    if not pct_str:
        return 0
    try:
        val = float(pct_str.replace('%', '').replace(',', '.'))
    except Exception:
        return 0
    if val > 2.2:
        return 5
    if 1.8 <= val <= 2.1:
        return 4
    if 1.2 <= val <= 1.7:
        return 3
    if 0.8 <= val <= 1.1:
        return 2
    if val < 0.8:
        return 1
    return 0

def weight_score_from_pct_str(pct_str: str) -> int:
    """Scores:
    5: >=22%
    4: 16-21.9%
    3: 10-15.9%
    2: 5-9.9%
    1: <5%
    """
    if not pct_str:
        return 0
    try:
        val = float(pct_str.replace('%', '').replace(',', '.'))
    except Exception:
        return 0
    if val >= 22:
        return 5
    if 16 <= val <= 21.9:
        return 4
    if 10 <= val <= 15.9:
        return 3
    if 5 <= val <= 9.9:
        return 2
    if val < 5:
        return 1
    return 0

# -------------------- UI --------------------
st.sidebar.header('Options & Upload')
uploaded = st.sidebar.file_uploader('Upload Excel (.xlsx/.xls) or CSV', type=['xlsx', 'xls', 'csv'])
# allow user to pick column after uploading
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
use_llm = st.sidebar.checkbox('Enable Gemini 2.0 Flash LLM extraction (reads sentence columns)', value=True)
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

if not uploaded:
    st.info('Upload your file (CSV or Excel). The app will let you choose the text column.')
    st.stop()

# Read file robustly and ensure we get a DataFrame
try:
    if uploaded.name.lower().endswith('.csv'):
        # try utf-8, fallback to latin1
        try:
            df_raw = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            df_raw = pd.read_csv(uploaded, encoding='latin1')
    else:
        df_raw = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else 0)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

# Clean column names (no change to user data)
cols = list(df_raw.columns)
st.sidebar.write("Detected columns:")
st.sidebar.write(cols)

# Let user pick the text columns and drug_name column (optional)
default_text_col = 'abstract' if 'abstract' in cols else cols[0]
text_col = st.sidebar.selectbox('Select column containing abstracts/text', options=cols, index=cols.index(default_text_col))
drug_col = None
if 'drug_name' in cols:
    drug_col = 'drug_name'
else:
    # let user choose if they have a drug name column
    drug_col = st.sidebar.selectbox('Optional: select drug name column (or choose "None")',
                                    options=['None'] + cols, index=0)
    if drug_col == 'None':
        drug_col = None

# Prepare Gemini model once
model = configure_gemini(API_KEY) if (use_llm and GENAI_AVAILABLE) else None
if use_llm and not GENAI_AVAILABLE:
    st.sidebar.warning("google-generativeai package not installed or import failed; LLM disabled. Install package to enable LLM.")

# -------------------- Process dataframe with regex (same logic) --------------------
@st.cache_data
def process_df_regex_only(df, text_column, drug_column):
    rows = []
    for _, row in df.iterrows():
        raw_text = row.get(text_column, '')
        # ensure cleaned text used for regex and sentence column
        text = clean_text_artifacts(raw_text if isinstance(raw_text, str) else str(raw_text or ""))
        # Extract
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')
        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')

        # HbA1c strict: keep only values < 7 OR from-to deltas (allow from-to irrespective)
        def allowed_hba(m):
            rp = m.get('reduction_pp')
            if 'from-to' in (m.get('type') or ''):
                return True
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                return float(abs(rp)) < 7.0
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

        new = row.to_dict()
        new.update({
            'sentence': ' | '.join(hba_sentences) if hba_sentences else '',
            'extracted_matches': [fmt_extracted_hba(m) for m in hba_matches],
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
            'weight_sentence': ' | '.join(wt_sentences) if wt_sentences else '',
            'weight_extracted_matches': [fmt_extracted_wt(m) for m in wt_matches],
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
        })
        rows.append(new)
    out = pd.DataFrame(rows)

    # Row keep rule: keep if either hba matches or weight matches
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

out_df = process_df_regex_only(df_raw, text_col, drug_col)

# -------------------- LLM extraction using the SENTENCE columns --------------------
# the LLM will read out_df['sentence'] and out_df['weight_sentence'] and extract reductions only.

if use_llm:
    # ensure series exist
    out_df['sentence'] = out_df.get('sentence', '').fillna('').astype(str)
    out_df['weight_sentence'] = out_df.get('weight_sentence', '').fillna('').astype(str)

    # helper that passes drug name if available in row
    def row_llm_hba(args):
        idx, s = args
        # pass drug name context if present
        drug = None
        try:
            drug = out_df.at[idx, drug_col] if (drug_col and drug_col in out_df.columns) else None
        except Exception:
            drug = None
        return llm_extract_from_sentence(model, "HbA1c", s, drug)

    def row_llm_wt(args):
        idx, s = args
        drug = None
        try:
            drug = out_df.at[idx, drug_col] if (drug_col and drug_col in out_df.columns) else None
        except Exception:
            drug = None
        return llm_extract_from_sentence(model, "Body weight", s, drug)

    # apply row-wise; keep it simple synchronous (Gemini calls per row)
    hba_results = []
    wt_results = []
    for i, r in out_df.iterrows():
        sent = clean_text_artifacts(str(r.get('sentence', '') or ''))
        w_sent = clean_text_artifacts(str(r.get('weight_sentence', '') or ''))
        h_vals, h_sel = ([], "")
        w_vals, w_sel = ([], "")
        if model is not None:
            try:
                if sent:
                    h_vals, h_sel = llm_extract_from_sentence(model, "HbA1c", sent, r.get(drug_col) if drug_col else None)
                if w_sent:
                    w_vals, w_sel = llm_extract_from_sentence(model, "Body weight", w_sent, r.get(drug_col) if drug_col else None)
            except Exception:
                h_vals, h_sel = [], ""
                w_vals, w_sel = [], ""
        # Normalize selected % to be positive (absolute)
        h_sel_norm = abs_pct_str(h_sel) if h_sel else ''
        w_sel_norm = abs_pct_str(w_sel) if w_sel else ''

        hba_results.append((h_vals, h_sel_norm))
        wt_results.append((w_vals, w_sel_norm))

    out_df["LLM extracted"] = [vals for vals, sel in hba_results]
    out_df["selected %"]   = [sel  for vals, sel in hba_results]
    out_df["Weight LLM extracted"] = [vals for vals, sel in wt_results]
    out_df["Weight selected %"]    = [sel  for vals, sel in wt_results]
else:
    # LLM disabled: leave columns empty
    out_df["LLM extracted"] = [[] for _ in range(len(out_df))]
    out_df["selected %"]   = ['' for _ in range(len(out_df))]
    out_df["Weight LLM extracted"] = [[] for _ in range(len(out_df))]
    out_df["Weight selected %"]    = ['' for _ in range(len(out_df))]

# -------------------- Scoring columns (beside selected %) --------------------
# place scores next to selected columns
out_df["A1c Score"] = out_df["selected %"].apply(lambda s: a1c_score_from_pct_str(s))
out_df["Weight Score"] = out_df["Weight selected %"].apply(lambda s: weight_score_from_pct_str(s))

# -------------------- Reorder columns so LLM columns and scores sit beside regex columns ----
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

# drop debug columns unless requested
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

# -------------------- Show results --------------------
st.write("### Results (first 200 rows shown)")
st.dataframe(display_df.head(200))

counts = out_df.attrs.get('counts', None)
if counts:
    kept, total = counts['kept'], counts['total']
    st.caption(
        f"Kept {kept} of {total} rows ({(kept/total if total else 0):.1%}).  "
        f"HbA1c-only: {counts['hba_only']}, Weight-only: {counts['wt_only']}, Both: {counts['both']}"
    )

# -------------------- Download cleaned results --------------------
@st.cache_data
def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

st.download_button(
    'Download results as Excel',
    data=to_excel_bytes(display_df),
    file_name='results_with_llm_and_scores.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

st.markdown('---')
st.write("**Notes & rules enforced:**")
st.write("- Regex extracts HbA1c reductions: whole-sentence from→to (delta) or % near HbA1c term within ±5 spaces. Weight has a previous-60-char fallback.")
st.write("- LLM reads the `sentence` / `weight_sentence` strings and extracts **only reported reductions** (ignores thresholds and inclusion criteria).")
st.write("- LLM prefers 12-month / T12 values over 6-month / T6 when both appear.")
st.write("- `selected %` columns are absolute (always positive).")
st.write("- `A1c Score` and `Weight Score` computed from `selected %` using the ranges you specified.")
