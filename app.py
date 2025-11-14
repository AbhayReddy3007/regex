# streamlit_hba1c_weight_llm_full.py
import re
import math
import json
from io import BytesIO
from typing import Tuple, List, Optional

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
API_KEY = ""  # <-- PUT YOUR GEMINI KEY HERE (e.g. "AIza...") - DO NOT COMMIT TO VCS
# =========================================================================

# Lazy import Gemini (app still runs without it; LLM columns will be empty)
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (sentence-driven extraction)")

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
def clean_unicode_artifacts(text: str) -> str:
    """Replace common garbled characters and normalize punctuation to ascii-friendly forms."""
    if not isinstance(text, str):
        return ''
    # normalize weird middle dot and broken sequences
    text = text.replace('\u00b7', '.')  # middle dot
    text = text.replace('Â·', '.')       # broken encoding
    text = text.replace('\u2013', '-')   # en dash
    text = text.replace('\u2014', '-')   # em dash
    text = text.replace('\xa0', ' ')     # non-breaking space
    # some CSVs show weird sequences — replace known ones
    text = text.replace('â‰¥', '>=')
    text = text.replace('â', '')
    # fix multiples of weird punctuation
    text = text.replace('·', '.')
    return text

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

# -------------------- Core extraction --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    """
    Within a qualifying sentence:
      • Scan the WHOLE SENTENCE for 'from X% to Y%' and record deltas.
      • For all other % patterns, search ONLY within a ±5-SPACES (inclusive-token) window
        around each term occurrence.
      • EXTRA (weight only): If window yields nothing, capture the NEAREST PREVIOUS % within 60 chars.
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
        # use gemini-2.0-flash as requested
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
    "You are an extractor with medical domain common-sense. Read the SENTENCE(S) provided and:\n"
    "1) Extract ONLY percentage changes that are *reductions* for the TARGET (either HbA1c/A1c or body weight).\n"
    "2) Prefer within-group/raw percent value mentioned next to the target (e.g., '13.88% body weight reduction'),\n"
    "   and do NOT pick between-group differences unless the sentence only reports between-group differences AND\n"
    "   no within-group numbers exist for that target in the same sentence. (So prefer 13.88% over -13.46% in the example.)\n"
    "3) If multiple drugs are mentioned and a 'drug_name' is provided, choose values that refer to that drug.\n"
    "4) If multiple timepoints exist, prefer 12 months / T12 over 6 months / T6.\n"
    "5) If 'from X% to Y%' appears, compute the delta (X - Y) and include it as a numeric percent (e.g. 'from 8% to 6%'=> '2%').\n"
    "6) Return STRICT JSON only, nothing else: {\"extracted\": [\"1.23%\",\"0.85%\"], \"selected_percent\": \"1.23%\"}.\n"
    "7) If no valid extraction for the TARGET exists, return {\"extracted\": [], \"selected_percent\": \"\"}.\n"
)

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name: Optional[str] = None) -> Tuple[List[str], str]:
    """
    Returns (extracted_list, selected_percent_str)
    If model is None or fails, returns ([], "")
    """
    if model is None or not sentence or sentence.strip() == "":
        return [], ""

    prompt = (
        f"TARGET: {target_label}\n"
        f"{'DRUG: ' + drug_name + '\\n' if drug_name else ''}"
        f"{LLM_RULES}\n\n"
        f"SENTENCE(S):\n{sentence}\n"
    )

    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # extract JSON substring
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e > s:
            obj = json.loads(text[s:e+1])
            extracted = [_norm_percent(x) for x in (obj.get("extracted") or []) if isinstance(x, str)]
            selected = _norm_percent(obj.get("selected_percent", "") or "")
            # make selected positive and normalized
            if selected:
                # strip sign
                sel_num = re.sub(r'[^\d.,-]', '', selected)
                try:
                    sel_val = parse_number(sel_num)
                    if not math.isnan(sel_val):
                        sel_val = abs(sel_val)
                        selected = fmt_pct(sel_val)
                except:
                    # keep as-is normalized string
                    pass
            return extracted, selected
    except Exception:
        pass

    return [], ""

# -------------------- Scoring maps --------------------
def a1c_score_from_pct(pct_float: Optional[float]) -> Optional[int]:
    """Mapping for A1c scores:
       5: >=2.2%
       4: 1.8-2.1%
       3: 1.2-1.7%
       2: 0.8-1.1%
       1: <0.8%
    """
    if pct_float is None or math.isnan(pct_float):
        return None
    if pct_float >= 2.2:
        return 5
    if 1.8 <= pct_float <= 2.1:
        return 4
    if 1.2 <= pct_float <= 1.7:
        return 3
    if 0.8 <= pct_float <= 1.1:
        return 2
    if pct_float < 0.8:
        return 1
    return None

def weight_score_from_pct(pct_float: Optional[float]) -> Optional[int]:
    """Mapping for weight scores:
       5: >=22%
       4: 16-21.9%
       3: 10-15.9%
       2: 5-9.9%
       1: <5%
    """
    if pct_float is None or math.isnan(pct_float):
        return None
    if pct_float >= 22:
        return 5
    if 16 <= pct_float <= 21.9:
        return 4
    if 10 <= pct_float <= 15.9:
        return 3
    if 5 <= pct_float <= 9.9:
        return 2
    if pct_float < 5:
        return 1
    return None

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
drug_name_col = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)
use_llm = st.sidebar.checkbox('Enable Gemini 2.0 Flash LLM extraction', value=False)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Choose the text column (default "abstract").')
    st.stop()

# read file robustly with utf-8 and fallback
try:
    if uploaded.name.lower().endswith('.csv'):
        # try utf-8, fallback to latin1
        try:
            df = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            df = pd.read_csv(uploaded, encoding='latin1', errors='replace')
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else 0)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

# configure gemini model (do not pass model into cached function to avoid hashing issue)
model = configure_gemini(API_KEY) if (use_llm and API_KEY and GENAI_AVAILABLE) else None
if use_llm and not GENAI_AVAILABLE:
    st.warning("google.generativeai not installed / not available — LLM columns will be empty.")
if use_llm and GENAI_AVAILABLE and not API_KEY:
    st.warning("You enabled LLM but API_KEY is empty. Fill API_KEY in the script to enable LLM.")

st.success(f'Loaded {len(df)} rows. Processing...')

# -------------------- PROCESSING (no streamlit cache for model safety) --------------------
def process_df(df_in: pd.DataFrame, text_col: str, model) -> pd.DataFrame:
    rows = []
    for idx, row in df_in.iterrows():
        raw_text = row.get(text_col, '')
        # clean text to remove encoding artifacts
        text = clean_unicode_artifacts(raw_text if isinstance(raw_text, str) else '')
        # run regex extraction
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')
        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')

        # HbA1c strict: keep only values < 7 OR from-to deltas
        def allowed_hba(m):
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                # allowed if reduction_pp < 7 (note reduction_pp can be delta)
                return abs(float(rp)) < 7.0
            for v in (m.get('values') or []):
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and abs(float(v)) < 7.0:
                        return True
                except Exception:
                    pass
            return False

        hba_matches = [m for m in hba_matches if allowed_hba(m)]
        # format the regex outputs
        def fmt_extracted_hba(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        wt_matches = wt_matches  # weight has no numeric cutoff
        def fmt_extracted_wt(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        # LLM extraction: read the *sentence* columns and extract values
        # Use drug_name if present to disambiguate
        drug_value = None
        if drug_name_col and drug_name_col in df_in.columns:
            dv = row.get(drug_name_col, None)
            if isinstance(dv, str) and dv.strip():
                drug_value = dv.strip()

        # Prepare sentence strings (join multiple sentences with ' | '), exactly what we show
        hba_sentence_str = " | ".join(hba_sentences) if hba_sentences else ""
        wt_sentence_str = " | ".join(wt_sentences) if wt_sentences else ""

        # Regex columns
        hba_regex_vals = [fmt_extracted_hba(m) for m in hba_matches]
        wt_regex_vals  = [fmt_extracted_wt(m) for m in wt_matches]

        # LLM columns (if model present)
        hba_llm_extracted, hba_selected = ([], "")
        wt_llm_extracted, wt_selected = ([], "")

        if model is not None:
            # HbA1c LLM
            hba_llm_extracted, hba_selected = llm_extract_from_sentence(model, "HbA1c", hba_sentence_str, drug_value)
            # Weight LLM
            wt_llm_extracted, wt_selected = llm_extract_from_sentence(model, "Body weight", wt_sentence_str, drug_value)

        # Ensure: if LLM extracted empty -> selected must be empty (do not fallback to regex)
        if not hba_llm_extracted:
            hba_selected = ""
        if not wt_llm_extracted:
            wt_selected = ""

        # Normalize selected % numeric values and make them positive
        def selected_to_numeric(sel_str: str) -> Optional[float]:
            if not sel_str or sel_str.strip() == "":
                return None
            s = sel_str.replace('%', '').replace(',', '.')
            try:
                v = float(re.sub(r'[^\d\.\-]', '', s))
                return abs(v)
            except Exception:
                return None

        hba_sel_num = selected_to_numeric(hba_selected)
        wt_sel_num  = selected_to_numeric(wt_selected)

        # Score columns
        a1c_score = a1c_score_from_pct(hba_sel_num)
        weight_score = weight_score_from_pct(wt_sel_num)

        # Put together final row
        new = row.to_dict()
        new.update({
            # sentences
            'sentence': hba_sentence_str,
            'weight_sentence': wt_sentence_str,
            # regex outputs
            'extracted_matches': hba_regex_vals,
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
            'weight_extracted_matches': wt_regex_vals,
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
            # LLM outputs
            'LLM extracted': hba_llm_extracted,
            'selected %': (fmt_pct(hba_sel_num) if hba_sel_num is not None else ""),
            'A1c Score': a1c_score,
            'Weight LLM extracted': wt_llm_extracted,
            'Weight selected %': (fmt_pct(wt_sel_num) if wt_sel_num is not None else ""),
            'Weight Score': weight_score,
        })
        rows.append(new)

    out_df = pd.DataFrame(rows)

    # Row keep rule: keep if HbA1c qualifies OR Weight qualifies (based on regex) — same as before
    def _list_has_items(x):
        return isinstance(x, list) and len(x) > 0
    mask_hba = (out_df['sentence'].astype(str).str.len() > 0) & (out_df['extracted_matches'].apply(_list_has_items))
    mask_wt  = (out_df['weight_sentence'].astype(str).str.len() > 0) & (out_df['weight_extracted_matches'].apply(_list_has_items))
    mask_keep = mask_hba | mask_wt
    out_df = out_df[mask_keep].reset_index(drop=True)

    # store counts
    out_df.attrs['counts'] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum()),
    )
    return out_df

out_df = process_df(df, col_name, model)

# -------------------- Reorder columns: place LLM columns BESIDE regex columns --------------------
def insert_after(cols: List[str], after: str, names: List[str]) -> List[str]:
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
def to_excel_bytes(df_to_save: pd.DataFrame) -> bytes:
    # ensure cleaned strings (no weird artifacts)
    df_copy = df_to_save.copy()
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = df_copy[col].apply(lambda x: clean_unicode_artifacts(str(x)) if pd.notna(x) else x)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_copy.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(display_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_llm_from_sentence.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# -------------------- Notes --------------------
st.markdown("""
**Notes & behavior**
- The LLM reads the `sentence` (for HbA1c) and `weight_sentence` (for weight) columns and extracts percentages for that target.
- If LLM finds nothing, `selected %` remains empty (we do NOT fallback to regex values).
- The app normalizes common encoding artifacts (so Excel output won't contain `Â·` etc).
- If multiple drugs appear, provide `drug_name` column name in the sidebar; the LLM will prefer values for that drug.
- Timepoint preference: T12/12 months > T6/6 months.
""")
