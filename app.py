# streamlit_hba1c_weight_llm.py
import re
import math
import json
import unicodedata
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== PUT YOUR GEMINI KEY HERE (HARD-CODE) =====================
# ⚠️ For local testing only. Do NOT commit this file with a real key to source control.
API_KEY = ""   # <- paste your Gemini API key here (e.g. "ABCD...") if you want LLM assistance
# ==============================================================================

# Lazy import of Gemini client so app still runs without it
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — Regex extractor + Gemini (reads sentence columns)")

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

def normalize_text_mojibake(s: str) -> str:
    """Fix common mojibake sequences and normalize unicode."""
    if not isinstance(s, str):
        return s
    # Replace common broken sequences seen from CSV/Excel exports
    s = s.replace('Â·', '·').replace('â‰¥', '≥').replace('â€“', '–').replace('â€”', '—') \
         .replace('Ã©', 'é').replace('Ã±', 'ñ').replace('\ufeff', '')
    # Normalize other unicode
    s = unicodedata.normalize('NFKC', s)
    # Remove weird control characters
    s = ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C' or ch in '\n\t\r')
    return s

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

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
use_llm    = st.sidebar.checkbox('Use Gemini LLM (Gemini 2.0 Flash)', value=True)
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
    st.stop()

# -------------------- Robust file reading --------------------
def read_uploaded(u):
    name = u.name.lower()
    # try utf-8 read first; if failure, fallback broadly
    if name.endswith('.csv'):
        try:
            df = pd.read_csv(u, encoding='utf-8')
        except Exception:
            try:
                df = pd.read_csv(u, encoding='latin1')
            except Exception:
                df = pd.read_csv(u, encoding='utf-8', errors='replace')
    else:
        # Excel: if user left sheet_name blank, read first sheet
        try:
            if sheet_name:
                df = pd.read_excel(u, sheet_name=sheet_name)
            else:
                df = pd.read_excel(u, sheet_name=0)
        except Exception:
            # final fallback: read all sheets and take first
            try:
                all_sheets = pd.read_excel(u, sheet_name=None)
                # all_sheets is dict of name -> df
                first_key = list(all_sheets.keys())[0]
                df = all_sheets[first_key]
            except Exception as e:
                raise
    # sanitize columns: if pandas returns a dict by mistake, convert
    if isinstance(df, dict):
        # convert to DataFrame using first sheet found
        first_key = list(df.keys())[0]
        df = df[first_key]
    return df

try:
    df = read_uploaded(uploaded)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

# normalize text columns to remove mojibake and trim whitespace
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].fillna('').astype(str).apply(normalize_text_mojibake)

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

st.success(f'Loaded {len(df)} rows and {len(df.columns)} columns. Processing...')

# -------------------- Gemini client configuration --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

model = configure_gemini(API_KEY) if use_llm else None
if use_llm and model is None:
    st.sidebar.warning("Gemini not available or API key missing/invalid — proceeding with regex-only outputs.")

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v

# LLM instruction rules (strict JSON output)
LLM_RULES = (
    "You are a strict extractor. INPUT: a single sentence (or pipe-separated sentences) and the TARGET (HbA1c or Body weight).\n"
    "OUTPUT: JSON only with keys {\"extracted\": [\"x%\",\"y%\"], \"selected_percent\": \"z%\"}.\n"
    "RULES:\n"
    " - Extract ONLY percentage values that represent a reported REDUCTION for the TARGET.\n"
    " - Ignore threshold/criteria mentions (e.g., '>=5%', 'HbA1c < 7.0%') — these are endpoints, not reductions.\n"
    " - If you see 'from X% to Y%' where that refers to the TARGET's measurement, compute delta = X - Y and include delta as a percentage (e.g. X - Y = 1.2%).\n"
    " - If multiple values exist at different timepoints, prefer values for 12 months / T12 over 6 months / T6. Recognize '12 months', '12-mo', 'T12', 'week 52' etc.\n"
    " - If the sentence mentions a drug name and the row provides a 'drug_name' column, prefer values that mention that drug (e.g., 'semaglutide yielded ...').\n"
    " - selected_percent must be ONE of the extracted list (choose the single best value). Return an empty string if none apply.\n"
    " - Use percent sign in outputs (e.g., '1.23%'). No extra commentary or text.\n"
)

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name: str = ""):
    """Call Gemini to extract target percentages from the given sentence(s).
       Returns (list_of_str_percent, selected_percent_str)."""
    if model is None or not sentence.strip():
        return [], ""
    prompt = (
        f"TARGET: {target_label}\n"
        f"DRUG_NAME: {drug_name}\n\n"
        f"{LLM_RULES}\n\n"
        f"SENTENCE:\n{sentence}\n"
    )
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = [ _norm_percent(x) for x in (data.get("extracted") or []) if isinstance(x, str) ]
            selected = _norm_percent(data.get("selected_percent", "") or "")
            # ensure selected is one of extracted if extracted present, else leave selected blank
            if extracted and selected and selected not in extracted:
                # allow LLM to propose cleaned value not in list — accept it but append into extracted
                extracted = extracted + [selected]
            return extracted, selected
    except Exception:
        # swallow LLM exceptions, will fallback to regex
        pass
    return [], ""

# -------------------- Processing pipeline --------------------
@st.cache_data
def process_df(df_in, text_col, model):
    rows = []
    # detect drug_name column if present
    drug_col = None
    for cand in ['drug_name', 'drug', 'treatment', 'intervention']:
        if cand in df_in.columns:
            drug_col = cand
            break

    for _, row in df_in.iterrows():
        text = row.get(text_col, '') or ''
        text = normalize_text_mojibake(text)

        # regex extraction
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')

        # STRICT FILTER for HbA1c: keep only values < 7 (prefer reduction_pp if present)
        def allowed_hba(m):
            rp = m.get('reduction_pp')
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

        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')

        def fmt_extracted_wt(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        hba_regex_vals = [fmt_extracted_hba(m) for m in hba_matches]
        wt_regex_vals = [fmt_extracted_wt(m) for m in wt_matches]

        # LLM extraction reading the sentence column (if any)
        sentence_field = 'sentence'  # we create this column from regex sentences below
        weight_sentence_field = 'weight_sentence'

        # Build sentence strings (joined) as the LLM input
        hba_sentence_str = ' | '.join(hba_sentences) if hba_sentences else ''
        wt_sentence_str  = ' | '.join(wt_sentences) if wt_sentences else ''

        # prefer drug_name if available
        drug_name_val = row.get(drug_col, '') if drug_col else ''

        # LLM extraction (if model available)
        hba_llm_vals, hba_selected = ([], "")
        wt_llm_vals, wt_selected = ([], "")

        if model is not None:
            if hba_sentence_str:
                hba_llm_vals, hba_selected = llm_extract_from_sentence(model, "HbA1c", hba_sentence_str, str(drug_name_val))
            if wt_sentence_str:
                wt_llm_vals, wt_selected = llm_extract_from_sentence(model, "Body weight", wt_sentence_str, str(drug_name_val))

        # If LLM produced nothing, fallback to regex candidates
        if not hba_llm_vals and hba_regex_vals:
            hba_llm_vals = hba_regex_vals[:]
        if not wt_llm_vals and wt_regex_vals:
            wt_llm_vals = wt_regex_vals[:]

        # Normalize and pick selected% fallback rules
        # Convert selected to absolute numeric (positive)
        def parse_and_abs_percent(s):
            if not s:
                return None
            s = s.replace('%', '').replace(' ', '').replace('·', '.').replace(',', '.')
            try:
                v = float(s)
                return abs(v)
            except Exception:
                return None

        # If LLM didn't choose a selected, try to pick best: prefer from-to delta, then largest absolute percent, then first
        if not hba_selected and hba_llm_vals:
            # try find delta-form in regex matches
            delta_candidates = [parse_and_abs_percent(x) for x in hba_llm_vals if 'from' in str(x).lower() or '-' in str(x)]
            delta_candidates = [x for x in delta_candidates if x is not None]
            if delta_candidates:
                best = delta_candidates[0]
                hba_selected = fmt_pct(best)
            else:
                nums = [parse_and_abs_percent(x) for x in hba_llm_vals]
                nums = [x for x in nums if x is not None]
                if nums:
                    best = max(nums)
                    hba_selected = fmt_pct(best)

        if not wt_selected and wt_llm_vals:
            delta_candidates = [parse_and_abs_percent(x) for x in wt_llm_vals if 'from' in str(x).lower() or '-' in str(x)]
            delta_candidates = [x for x in delta_candidates if x is not None]
            if delta_candidates:
                best = delta_candidates[0]
                wt_selected = fmt_pct(best)
            else:
                nums = [parse_and_abs_percent(x) for x in wt_llm_vals]
                nums = [x for x in nums if x is not None]
                if nums:
                    best = max(nums)
                    wt_selected = fmt_pct(best)

        # convert selected to positive string (already abs above). If still missing, set empty
        if hba_selected:
            # ensure percent formatting
            vnum = parse_and_abs_percent(hba_selected)
            hba_selected = fmt_pct(vnum) if vnum is not None else ''
        if wt_selected:
            vnum = parse_and_abs_percent(wt_selected)
            wt_selected = fmt_pct(vnum) if vnum is not None else ''

        new = row.to_dict()
        # HbA1c columns
        new.update({
            'sentence': hba_sentence_str,
            'extracted_matches': hba_regex_vals,
            'LLM extracted': hba_llm_vals,
            'selected %': hba_selected,
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
        })
        # Weight columns
        new.update({
            'weight_sentence': wt_sentence_str,
            'weight_extracted_matches': wt_regex_vals,
            'Weight LLM extracted': wt_llm_vals,
            'Weight selected %': wt_selected,
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
        })

        rows.append(new)

    out = pd.DataFrame(rows)

    # -------- Row keeping rule: keep if HbA1c qualifies OR Weight qualifies --------
    def _list_has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(_list_has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(_list_has_items))
    mask_keep = mask_hba | mask_wt
    out = out[mask_keep].reset_index(drop=True)

    # Add quick counts for sanity check
    out.attrs['counts'] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum()),
    )

    return out

# Process
out_df = process_df(df, col_name, model)

# -------------------- Score calculation --------------------
def a1c_score_from_selected_pct(pct_str):
    """Scores for A1c:
    5: >2.2%
    4: 1.8%-2.1%
    3: 1.2%-1.7%   (note: you had a formatting typo — I'm using 1.2-1.7)
    2: 0.8%-1.1%
    1: <0.8%
    """
    if not pct_str:
        return ''
    s = pct_str.replace('%', '').replace('·', '.').replace(',', '.').strip()
    try:
        v = float(s)
    except:
        return ''
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
    return ''

def weight_score_from_selected_pct(pct_str):
    """Scores for weight:
    5: >=22%
    4: 16-21.9%
    3: 10-15.9%
    2: 5-9.9%
    1: <5%
    """
    if not pct_str:
        return ''
    s = pct_str.replace('%', '').replace('·', '.').replace(',', '.').strip()
    try:
        v = float(s)
    except:
        return ''
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
    return ''

# Insert score columns beside selected % columns
out_df['A1c Score'] = out_df['selected %'].apply(a1c_score_from_selected_pct)
out_df['Weight Score'] = out_df['Weight selected %'].apply(weight_score_from_selected_pct)

# Ensure selected % columns are strings (no negative signs) and Weight selected % etc are positive
def normalize_selected_col(s):
    if not s: return ''
    s2 = s.replace('%', '').replace('·', '.').replace(',', '.').strip()
    try:
        v = abs(float(s2))
        return fmt_pct(v)
    except:
        return s

out_df['selected %'] = out_df['selected %'].apply(normalize_selected_col)
out_df['Weight selected %'] = out_df['Weight selected %'].apply(normalize_selected_col)

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
# reorder only if those columns exist
display_df = display_df[[c for c in cols if c in display_df.columns]]

# -------------------- Show results --------------------
st.write("### Results (first 200 rows shown)")
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

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
    # make sure Excel-friendly encoding and no mojibake in cells
    df_to_save = df.copy()
    for c in df_to_save.columns:
        if df_to_save[c].dtype == object:
            df_to_save[c] = df_to_save[c].astype(str).apply(normalize_text_mojibake)
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_to_save.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(display_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_llm_from_sentence.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# -------------------- Notes / Quick help --------------------
st.markdown("""
**Notes & behavior**
- The LLM reads only the `sentence` (HbA1c) and `weight_sentence` (weight) columns and extracts reductions only.
- Threshold phrases like `>=5%` or `HbA1c < 7.0%` are recognized as endpoints and ignored by the LLM extraction rules.
- If a `drug_name` column exists, the LLM will prefer values mentioning the drug.
- If Gemini is unavailable or you leave `API_KEY` empty, the app falls back to regex-only extraction.
- `selected %` columns are forced positive (absolute value).
""")
