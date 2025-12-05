# streamlit_hba1c_weight_llm.py
import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
API_KEY = ""   # <- replace with your real key
# =========================================================================

# Baseline weight for converting kg -> percent
BASELINE_WEIGHT = 105.0  # kg (change if needed)

# Baseline A1c for relative reduction when only target threshold is reported
BASELINE_A1C = 8.2  # %

# Lazy Gemini import so the app still runs without the package
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (reads the sentence column)")

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
# units for weight (kg/kilogram)
re_weight_unit = re.compile(r'\bkg\b|\bkilograms?\b', FLAGS)

# Pattern for “achieved HbA1c/A1c ≤ x% / < x%” (used with baseline A1c)
THRESH_A1C_RE = re.compile(
    r'\b(?:achiev(?:e|ed|ing|ement)?|attain(?:ed|ment)?|target)\b[^.]*?'
    r'\b(?:hb\s*a1c|hba1c|a1c)\b[^<%]*?(?:≤|<=|<)\s*([+-]?\d+(?:[.,·]\d+)?)\s*%',
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
    parts = re.split(r'(?<=[\.!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    """Require: (target term) AND a % (or kg for weight) AND a reduction cue."""
    has_term = bool(term_re.search(sent))
    has_pct = bool(re_pct.search(sent))
    has_kg = bool(re_weight_unit.search(sent))
    has_cue = bool(re_reduction_cue.search(sent))

    # If target is weight, allow either percent OR kg as unit for extraction
    if term_re == re_weight:
        return has_term and (has_pct or has_kg) and has_cue
    # otherwise (e.g., HbA1c) require a percent
    return has_term and has_pct and has_cue

def fmt_pct(v):
    """Format percent with 2 decimal places (strip trailing zeros if any)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.2f}".rstrip('0').rstrip('.')
    return f"{s}%"

# Duration extraction helper (regex baseline)
DURATION_RE = re.compile(
    r'\b(?:T\d{1,2}|'
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:weeks?|wks?|wk|w)\b|'
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:months?|mos?|mo)\b|'
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:days?|d)\b|'
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:years?|yrs?)\b|'
    r'\d{1,3}-week\b|\d{1,3}-month\b|\d{1,3}-mo\b)',
    FLAGS
)

def extract_durations_regex(text: str) -> str:
    """Regex-based fallback duration extractor."""
    if not isinstance(text, str) or not text.strip():
        return ""
    found = []
    seen = set()
    for m in DURATION_RE.finditer(text):
        token = m.group(0).strip()
        # normalize whitespace and hyphens
        token = re.sub(r'\s+', ' ', token)
        token = token.replace('–', '-').replace('—', '-')
        # normalize common abbreviations
        token = re.sub(r'\bmos?\b', 'months', token, flags=re.IGNORECASE)
        token = re.sub(r'\bmo\b', 'months', token, flags=re.IGNORECASE)
        token = re.sub(r'\bwks?\b', 'weeks', token, flags=re.IGNORECASE)
        token = re.sub(r'\bw\b', 'weeks', token, flags=re.IGNORECASE)
        token = re.sub(r'\bd\b', 'days', token, flags=re.IGNORECASE)
        token = re.sub(r'\byrs?\b', 'years', token, flags=re.IGNORECASE)
        token = token.strip()
        if token.lower() not in seen:
            seen.add(token.lower())
            found.append(token)
    return ' | '.join(found)

# --- build a local window by counting spaces (and INCLUDE bordering tokens) ---
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

    # Clamp
    start = max(0, min(start, L))
    end = max(start, min(end if end != pos else L, L))
    return s[start:end], start, end

def add_match(out, si, abs_start, m, typ, values, reduction):
    out.append({
        'raw': m.group(0) if hasattr(m, 'group') else str(m),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (abs_start + (m.start() if hasattr(m, 'start') else 0), abs_start + (m.end() if hasattr(m, 'end') else 0)),
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

    # 1) WHOLE-SENTENCE: from X% to Y%  -> absolute pp AND relative % using ((X - Y) / Y) * 100 per request
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red_pp = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        # compute relative reduction percent using ((a - b) / b) * 100 if possible (guard b != 0)
        rel = None
        if not (math.isnan(a) or math.isnan(b)) and b != 0:
            rel = ((a - b) / b) * 100.0
        # add the absolute-from-to match (keep for completeness)
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red_pp)
        # ALSO compute relative reduction % = ((a - b) / b) * 100  (if b != 0)
        if rel is not None:
            rel_raw = f"{rel:.6f}%"  # keep higher precision here; formatting is applied later
            matches.append({
                'raw': rel_raw,
                'type': f'{tag_prefix}:from-to_relative_percent',
                'values': [a, b, rel],
                'reduction_pp': red_pp,
                'sentence_index': si,
                'span': (m.start(), m.end()),
            })

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

        # NEW: if weight tag, also look for kg values in the window
        if tag_prefix == 'weight':
            for m in re.finditer(r'([+-]?\d+(?:[.,·]\d+)?)\s*(?:kg|kilograms?)', seg, FLAGS):
                any_window_hit = True
                v = parse_number(m.group(1))
                add_match(matches, si, abs_s, m, f'{tag_prefix}:kg_pmSpaces5', [v, 'kg'], v)

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
drug_col   = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)
use_llm   = st.sidebar.checkbox('Enable Gemini LLM (Gemini 2.0 Flash)', value=True)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
    st.stop()

# read file robustly and handle common encodings
try:
    if uploaded.name.lower().endswith('.csv'):
        try:
            df = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            # fallback: replace bad characters
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding='utf-8', error_bad_lines=False, engine='python')
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else None)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

if drug_col and drug_col not in df.columns:
    st.warning(f'Drug-name column "{drug_col}" not found. Drug-based disambiguation will be skipped.')

st.success(f'Loaded {len(df)} rows. Processing...')

# -------------------- Gemini 2.0 Flash setup --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

model = configure_gemini(API_KEY) if use_llm else None

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v

# -------------------- DETAILED LLM RULES --------------------
LLM_RULES = """
You are a focused information-extraction assistant. Read the provided SENTENCE(S) and extract **change magnitudes** that represent reported changes for the specified TARGET (either "HbA1c"/"A1c" or "Body weight"). Return strict JSON only (no explanation, no commentary, no extra text). Follow these rules exactly.

OUTPUT SPECIFICATION (exact JSON structure)
Return one JSON object. Allowed keys (you may include some or all as appropriate):
{
  "extracted": ["1.23%", "3.5 kg", "25.0% (relative)"],     // REQUIRED (array; may be empty). Include all plausible candidates you find in the sentence.
  "selected_percent": "1.23%",                              // OPTIONAL: best percent value (absolute or relative) — positive magnitude, trailing '%'
  "selected_relative_percent": "25.0%",                     // OPTIONAL: the relative % computed from a `from X% to Y%` phrase ((X - Y) / Y * 100)
  "selected_kg": "3.5 kg",                                  // OPTIONAL: best kg value (absolute, positive, suffix ' kg')
  "confidence": 0.92                                         // OPTIONAL: number 0.0–1.0 indicating your confidence (only if available)
}

KEY PRINCIPLES (strict)
1) JSON-only: return exactly one JSON object and nothing else. The object must be valid JSON parseable by a machine.

2) Normalization:
   - Decimal separators -> '.' (examples: '13·88' or '13,88' -> '13.88').
   - Percent strings MUST end with '%' and use '.' for decimals (e.g., "13.88%").
   - Kg strings MUST use the format "<number> kg" with a space and 'kg' suffix (e.g., "3.64 kg").
   - Remove typographic artifacts and extra whitespace.

3) What to extract:
   - Extract only **change magnitudes** (reported reductions or increases) relevant to the TARGET.
   - Accept these forms (non-exhaustive): "reduced by 1.5%", "decrease of 1.2%", "13.88% body weight reduction", "mean decrease of 3.64 kg", "reduced by 3.5 kg", "from 8.5% to 7.0%".
   - **Do NOT** extract thresholds, eligibility cutoffs or endpoints (examples: "≥5%", "HbA1c < 7.0%", "proportion achieving ≥5%"), SEs, CIs, p-values, sample-size percentages, or labels.
   - If a percent appears in parentheses and is clearly an SE/CI (e.g., "(SE 0.90)"), do NOT extract it.

4) `from X% to Y%` handling (absolute and relative):
   - When you detect a `from X% to Y%` pattern (or wording like "declined from X% to Y%"):
     a) Compute ABSOLUTE change in percentage points: `abs_pp = X - Y`.
     b) Compute RELATIVE reduction % using the formula: `((X - Y) / Y) * 100`.
        - Format as "N.N%" (dot decimal), round reasonably, include in `extracted` and set `selected_relative_percent` to this value if appropriate.
   - Normalization: compute numerically, round to sensible precision (2–3 decimals ok), and format as "N.N%".

5) Units to return:
   - Percent values: strings with '%' (e.g., "1.75%").
   - Kg values: strings with ' kg' suffix (e.g., "3.64 kg").
   - Return both types if both are present in the sentence (so the app can decide which to prefer).

6) Drug context (`DRUG_HINT`):
   - You will be given an optional `DRUG_HINT` string. Use it to disambiguate when multiple drugs/groups are mentioned.
   - Prefer values that are explicitly tied to the DRUG_HINT (e.g., "in the semaglutide group ... -1.75%").
   - If you cannot confidently tie a value to the DRUG_HINT, include candidates in `extracted` and leave `selected_*` empty.

7) Timepoint preference:
   - If multiple timepoints are present (e.g., T12, 12-mo, week 52 vs T6 or week 26), prefer values at **12 months** (T12) over 6 months (T6) and unspecified ones. When selecting the best value, prefer T12 if applicable.

8) Preference / selection rules (app-level guidance):
   - The app is configured to **prefer percent** for weight selection. So:
     • If a percent is present and clearly represents the weight change for the target/drug, set `selected_percent` to that percent (absolute magnitude, positive).
     • If no suitable percent exists but a kg change is present & tied to the target/drug, set `selected_kg` to that kg and also include it in `extracted`. The app may convert kg → percent later using a baseline weight.
   - Nevertheless, still return both percent and kg (if both exist) so the app has full context.

9) Sign handling:
   - If the sentence reports negative signs (e.g., "-3.5 kg", "-1.75%"), you may put negative values in `extracted` if you want, but **selected_*** fields must be positive absolute magnitudes (no leading '+' or '-') and must include units.

10) Ambiguity & fallbacks:
   - If multiple plausible change values are present and you can confidently pick one for the DRUG_HINT and timepoint, set `selected_*` accordingly.
   - If ambiguous (cannot tie to DRUG_HINT or cannot decide), include all plausible candidates in `extracted` and leave all `selected_*` fields empty.
   - If you cannot find any valid change magnitude, return `"extracted": []` and omit `selected_*` fields (or set them to empty strings).

11) Output strictness & validations:
   - JSON must parse. Allowed keys: `extracted` (required), `selected_percent` (optional), `selected_relative_percent` (optional), `selected_kg` (optional), `confidence` (optional).
   - `extracted` must be an array (can be `[]`). Each element must be a string with appropriate unit (e.g., "1.23%", "3.64 kg", "25.0% (relative)").
   - Do not print any extra commentary, logs, or explanation.

12) Formatting & rounding:
   - Use dot decimal. Reasonable rounding: 1–3 decimal places (keep precision, but avoid unnecessary trailing zeros; e.g., "3.5 kg", "1.75%").
   - For relative computations, 1–2 decimal places is fine (e.g., "25.0%").

13) Examples (behavioral guidance — follow these examples exactly):
   - Example A:
       Sentence: "At week 36, semaglutide yielded a 13·88% (SE 0·90) body weight reduction compared with 0·42% ... (between-group difference: -13·46%)."
       -> extracted should include "13.88%". You may include "13.46%" as additional candidate. selected_percent should be "13.88%".
   - Example B:
       Sentence: "Patients experienced mean decreases in HbA1c and weight from baseline to 6 months of -1.75% ... and -3.64 kg ... in the oral semaglutide group..."
       DRUG_HINT: "semaglutide"
       -> extracted should include "1.75%" and "3.64 kg"; selected_percent "1.75%" and selected_kg "3.64 kg" (if asked to pick both). If the app prefers percent, selected_percent must be set and used.
   - Example C:
       Sentence: "Declined from 8.0% to 6.0%."
       -> absolute change = 2.0 pp; relative reduction = ((8 - 6)/6)*100 = 33.333...%. Include "33.333% (relative)" in extracted and set selected_relative_percent "33.33%" if that relative value is the best representative percent for the sentence.
   - Example D:
       Sentence: "Mean reduction: 3,5 kg (p<0.01)."
       -> extracted should include "3.5 kg". Do NOT include "p<0.01" or anything related to p-value.

14) Edge-case rules:
   - If percent is attached to a threshold/target context (e.g., "proportion achieving ≥5%"), do NOT treat as a reduction.
   - If you see ranges like "1.0–1.5%" and it refers to reduction magnitude, you may include the range (normalized) or include the max value as candidate — be conservative; if unsure, include both ends or both as separate entries like "1.0%","1.5%" and select the highest entry.
   - If multiple drugs appear and DRUG_HINT is not provided, include candidates for all and leave selected_* empty.

15) Confidence (optional):
   - If you can compute a meaningful confidence (0–1) that the returned selected_* is correct, you may include `"confidence": <float>`.

16)- In cases like "Patients experienced mean decreases in HbA1C and weight from baseline to 6 months of -1.75% (P< 0.001) and -3.64 kg (P= 0.015), respectively, in the oral semaglutide group and -1.35% (P< 0.001) and -5.26 kg (P< 0.001)" always choose the HbA1c reduction associated with the highest weight loss. In this case, it is -1.35%

17) In cases like this "0.5 mg dose lowered HbA1c by 1.2-1.5 %, while the 1.0 mg dose reduced it by 1.4-1.8 %" always choose the value associated with the highest strength/dosage. In this case it is 1.8%.

19) In cases like this "Median weight reduced from 100.0 kg to 91.5 kg (p<0.001), and median BMI decreased from 33.6 to 30.9 kg/m² (p<0.001)." calculate the difference in kg first and then find the percentage. In this case, it is 8.5%. Selected % must be 8.5%

"""

# -------------------- LLM RULES FOR DURATION --------------------
LLM_DURATION_RULES = """
You are a precise information-extraction assistant. Read the full ABSTRACT and extract the study duration/timepoints.

Return JSON only, with this structure:

{
  "durations": ["24 weeks", "52 weeks", "T12", "12 months"],   // array of all relevant duration mentions
  "primary_duration": "52 weeks"                               // single best/primary follow-up duration
}

Rules:
- Consider phrases like: "24 weeks", "26 weeks", "52-week", "1 year", "2 years", "12 months", "6 months", "T6", "T12", "week 26", "week 52".
- Normalize:
  - Use "weeks", "months", "years" (e.g., "52 weeks", "12 months", "2 years").
  - Keep T-timepoints as "T6", "T12", etc.
- Include all relevant follow-up durations in "durations".
- For "primary_duration":
  - Prefer the *longest* clearly reported follow-up that corresponds to primary/endpoint results (e.g., 52 weeks > 26 weeks; 12 months > 6 months).
  - If multiple candidates have similar length, you may choose any that clearly reflects main follow-up.
- Do NOT include durations that are clearly not study follow-up (e.g., "washout for 3 days" if the main outcome is at 24 weeks).
- JSON only, no explanation.
"""

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_hint: str = ""):
    """
    Returns (extracted_list (strings like '1.23%' or '3.5 kg'), selected_percent (string like '1.23%'), selected_kg_or_rel (string))
    If model is None, returns ([], '', '')
    """
    if model is None or not sentence.strip():
        return [], "", ""

    prompt = (
        f"TARGET: {target_label}\n"
        + (f"DRUG_HINT: {drug_hint}\n" if drug_hint else "")
        + LLM_RULES + "\n"
        + "SENTENCE:\n" + sentence + "\n"
        + "Return JSON only.\n"
    )

    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # find JSON object in response
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = []
            is_hba_target = str(target_label).strip().lower().startswith('hb')
            for x in (data.get("extracted") or []):
                if not isinstance(x, str):
                    continue
                x = x.strip()
                # If target is HbA1c, ignore any kg items (do not extract kg for A1c)
                if is_hba_target and re.search(r'\bkg\b', x, re.IGNORECASE):
                    continue
                # kg values (allowed for weight target)
                if re.search(r'\bkg\b', x, re.IGNORECASE):
                    m = re.search(r'([+-]?\d+(?:[.,·]\d+)?)', x)
                    if m:
                        num = parse_number(m.group(1))
                        if not math.isnan(num):
                            s_num = f"{num:.3f}".rstrip('0').rstrip('.')
                            extracted.append(f"{s_num} kg")
                    continue
                # percent
                x2 = _norm_percent(x)
                if re.search(r'[<>≥≤]', x2):
                    continue
                if re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', x2):
                    extracted.append(_norm_percent(x2))
            # selected_percent if present
            selected = _norm_percent(data.get("selected_percent", "") or "")
            if selected:
                try:
                    num = parse_number(selected.replace('%', ''))
                    if not math.isnan(num):
                        num = abs(num)
                        selected = fmt_pct(num)
                except:
                    selected = ""

            selected_kg_or_rel = ""
            if 'selected_kg' in data and data.get('selected_kg'):
                sk = (data.get('selected_kg') or "").strip()
                m = re.search(r'([+-]?\d+(?:[.,·]\d+)?)', sk)
                if m:
                    num = parse_number(m.group(1))
                    if not math.isnan(num):
                        s_num = f"{num:.3f}".rstrip('0').rstrip('.')
                        selected_kg_or_rel = f"{s_num} kg"
            # fallback: selected_relative_percent
            if not selected_kg_or_rel and 'selected_relative_percent' in data and data.get('selected_relative_percent'):
                sr = _norm_percent(data.get('selected_relative_percent') or "")
                if sr:
                    selected_kg_or_rel = sr

            return extracted, selected, selected_kg_or_rel
    except Exception:
        return [], "", ""

    return [], "", ""

def llm_extract_duration(model, abstract_text: str, drug_hint: str = ""):
    """
    Use LLM to extract duration/timepoints from the FULL ABSTRACT.
    Returns (durations_list, primary_duration_str)
    """
    if model is None or not abstract_text.strip():
        return [], ""

    prompt = (
        "TARGET: Duration\n"
        + (f"DRUG_HINT: {drug_hint}\n" if drug_hint else "")
        + LLM_DURATION_RULES + "\n"
        + "ABSTRACT:\n" + abstract_text + "\n"
        + "Return JSON only.\n"
    )

    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            durations = []
            for x in (data.get("durations") or []):
                x = str(x).strip()
                if x:
                    durations.append(x)
            primary = (data.get("primary_duration") or "").strip()
            return durations, primary
    except Exception:
        return [], ""

    return [], ""

# -------------------- Scoring helpers --------------------
def compute_a1c_score(selected_pct_str: str):
    """Scores for A1c:
    5: >2.2%
    4: 1.8%-2.1%
    3: 1.2%-1.7%
    2: 0.8%-1.1%
    1: <0.8%
    """
    if not selected_pct_str:
        return ""
    try:
        val = parse_number(selected_pct_str.replace('%', ''))
    except:
        return ""
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
    return ""

def compute_weight_score(selected_pct_str: str):
    """Scores for weight:
    5: >=22%
    4: 16-21.9%
    3: 10-15.9%
    2: 5-9.9%
    1: <5%
    """
    if not selected_pct_str:
        return ""
    try:
        val = parse_number(selected_pct_str.replace('%', ''))
    except:
        return ""
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
    return ""

# -------------------- Processing function --------------------
@st.cache_data
def process_df(_model, df_in: pd.DataFrame, text_col: str, drug_col_name: str):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, '')
        if not isinstance(text_orig, str):
            text_orig = '' if pd.isna(text_orig) else str(text_orig)

        # Regex-based duration baseline
        duration_regex_str = extract_durations_regex(text_orig)

        # Run regex extraction
        hba_matches, hba_sentences = extract_sentences(text_orig, re_hba1c, 'hba1c')
        wt_matches, wt_sentences   = extract_sentences(text_orig, re_weight, 'weight')

        # ------------------ compute precomputed relative values (from regex) ------------------
        def _find_relative_fromto(matches):
            # prefer explicit relative match
            for m in matches:
                t = (m.get('type') or '').lower()
                if 'from-to_relative_percent' in t:
                    vals = m.get('values') or []
                    if len(vals) >= 3 and vals[2] is not None and not math.isnan(vals[2]):
                        return fmt_pct(vals[2])
            # fallback: compute from absolute from-to match if available using ((a - b)/b)*100
            for m in matches:
                t = (m.get('type') or '').lower()
                if 'from-to_sentence' in t:
                    vals = m.get('values') or []
                    if len(vals) >= 2:
                        a = vals[0]; b = vals[1]
                        if not (math.isnan(a) or math.isnan(b)) and b != 0:
                            rel = ((a - b) / b) * 100.0
                            return fmt_pct(rel)
            return None

        precomputed_hba_rel = _find_relative_fromto(hba_matches)
        precomputed_wt_rel  = _find_relative_fromto(wt_matches)
        # -------------------------------------------------------------------------------

        # Format regex outputs
        def fmt_extracted(m):
            t = (m.get('type') or '').lower()
            # Prefer relative from-to percent if present
            if 'from-to' in t:
                if 'relative' in t:
                    vals = m.get('values') or []
                    if len(vals) >= 3 and vals[2] is not None and not math.isnan(vals[2]):
                        return fmt_pct(vals[2])
                    if m.get('reduction_pp') is not None and not math.isnan(m.get('reduction_pp')):
                        return fmt_pct(m.get('reduction_pp'))
                    return m.get('raw', '')
                else:
                    # absolute from-to entry: try to return reduction_pp if available (but relative preferred)
                    if m.get('reduction_pp') is not None and not math.isnan(m.get('reduction_pp')):
                        return fmt_pct(m.get('reduction_pp'))
                    return m.get('raw', '')
            # other types
            if isinstance(m.get('reduction_pp'), (int, float)) and not math.isnan(m.get('reduction_pp')):
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        hba_regex_vals = [fmt_extracted(m) for m in hba_matches]
        wt_regex_vals  = [fmt_extracted(m) for m in wt_matches]

        # Build sentence strings (join qualifying sentences with ' | ')
        sentence_str = ' | '.join(hba_sentences) if hba_sentences else ''
        weight_sentence_str = ' | '.join(wt_sentences) if wt_sentences else ''

        # LLM extraction: read from the sentence column and produce LLM extracted + selected %
        drug_hint = ""
        if drug_col_name and drug_col_name in df_in.columns:
            drug_hint = str(row.get(drug_col_name, '') or "")

        # HbA1c LLM
        hba_llm_extracted, hba_selected, _ = ([], "", "")
        if _model is not None and sentence_str:
            hba_llm_extracted, hba_selected, _ = llm_extract_from_sentence(_model, "HbA1c", sentence_str, drug_hint)

        # Weight LLM
        wt_llm_extracted, wt_selected, wt_selected_fallback = ([], "", "")
        if _model is not None and weight_sentence_str:
            wt_llm_extracted, wt_selected, wt_selected_fallback = llm_extract_from_sentence(_model, "Body weight", weight_sentence_str, drug_hint)

        # --- FORCE precomputed relative into LLM outputs if LLM didn't pick one ---
        # For HbA1c: prefer precomputed relative (if present)
        if precomputed_hba_rel:
            if precomputed_hba_rel not in (hba_llm_extracted or []):
                hba_llm_extracted = ([precomputed_hba_rel] if not hba_llm_extracted else [precomputed_hba_rel] + hba_llm_extracted)
            if not hba_selected:
                hba_selected = precomputed_hba_rel

        # For Weight: do similarly but careful to prefer LLM percent candidates if present
        if precomputed_wt_rel:
            if precomputed_wt_rel not in (wt_llm_extracted or []):
                wt_llm_extracted = ([precomputed_wt_rel] if not wt_llm_extracted else [precomputed_wt_rel] + wt_llm_extracted)
            if not wt_selected:
                wt_selected = precomputed_wt_rel
        # ----------------------------------------------------------------

        # ALSO include any kg matches found by the regex into the weight LLM extracted list (but NOT into HbA1c)
        def _fmt_kg(v):
            try:
                fv = float(v)
            except:
                try:
                    fv = parse_number(v)
                except:
                    return None
            s = f"{fv:.3f}".rstrip('0').rstrip('.')
            return f"{s} kg"

        kg_values = []
        for m in wt_matches:
            t = (m.get('type') or '').lower()
            if ':kg' in t or 'kg' in t:
                vals = m.get('values') or []
                if vals:
                    v = vals[0]
                    kg_str = _fmt_kg(v)
                    if kg_str:
                        kg_values.append(kg_str)

        # ONLY attach kg_values to weight LLM extracted (do NOT attach kg to HbA1c LLM extracted)
        if kg_values:
            wt_llm_extracted = (wt_llm_extracted or []) + kg_values

        # ---- Baseline A1c threshold handling (achieved HbA1c/A1c <= x% or < x%) ----
        # Only apply if no A1c reduction % has been selected yet.
        if not hba_selected and BASELINE_A1C and not math.isnan(BASELINE_A1C):
            m_thresh = THRESH_A1C_RE.search(text_orig)
            if m_thresh:
                target_a1c = parse_number(m_thresh.group(1))
                if not math.isnan(target_a1c) and BASELINE_A1C != 0:
                    rel = ((BASELINE_A1C - target_a1c) / float(BASELINE_A1C)) * 100.0
                    rel_pct_str = fmt_pct(rel)
                    if not hba_llm_extracted:
                        hba_llm_extracted = []
                    if rel_pct_str not in hba_llm_extracted:
                        hba_llm_extracted = [rel_pct_str] + hba_llm_extracted
                    hba_selected = rel_pct_str
        # ---------------------------------------------------------------------------

        # If LLM extracted is empty, ensure selected remains empty (explicit requirement)
        if not hba_llm_extracted:
            hba_selected = ""
        if not wt_llm_extracted:
            wt_selected = ""

        # --- Selection logic for weight: pick HIGHEST weight-loss percentage overall ---
        def percent_str_to_raw(pct_str):
            if not pct_str:
                return None
            s = pct_str.replace('%', '').replace(',', '.').strip()
            try:
                return abs(float(s))
            except:
                return None

        def kg_to_percent_raw(kg_str):
            if not kg_str:
                return None
            m = re.search(r'([+-]?\d+(?:[.,·]\d+)?)', kg_str)
            if not m:
                return None
            num = parse_number(m.group(1))
            if math.isnan(num) or BASELINE_WEIGHT == 0:
                return None
            pct = (abs(num) / float(BASELINE_WEIGHT)) * 100.0
            return pct

        # Collect percent candidates (LLM selected percent + percent strings in LLM extracted)
        pct_candidates = []
        if wt_selected:
            n = percent_str_to_raw(wt_selected)
            if n is not None:
                pct_candidates.append((n, _norm_percent(wt_selected)))
        for item in (wt_llm_extracted or []):
            if isinstance(item, str) and item.strip().endswith('%'):
                n = percent_str_to_raw(item)
                if n is not None:
                    pct_candidates.append((n, _norm_percent(item.strip())))

        # Collect kg candidates and convert to percent
        kg_pct_candidates = []
        # LLM-provided kg in fallback
        if wt_selected_fallback and isinstance(wt_selected_fallback, str) and re.search(r'\bkg\b', wt_selected_fallback):
            p = kg_to_percent_raw(wt_selected_fallback)
            if p is not None:
                kg_pct_candidates.append((p, wt_selected_fallback))
        # kg strings in extracted list
        for item in (wt_llm_extracted or []):
            if isinstance(item, str) and re.search(r'\bkg\b', item):
                p = kg_to_percent_raw(item)
                if p is not None:
                    kg_pct_candidates.append((p, item))
        # regex-found kg_values
        for k in kg_values:
            p = kg_to_percent_raw(k)
            if p is not None and all(k != existing[1] for existing in kg_pct_candidates):
                kg_pct_candidates.append((p, k))

        # Now pick the highest percent among pct_candidates and kg_pct_candidates
        best_num = None
        if pct_candidates:
            pct_candidates.sort(key=lambda x: x[0], reverse=True)
            best_num = pct_candidates[0][0]
        if kg_pct_candidates:
            kg_pct_candidates.sort(key=lambda x: x[0], reverse=True)
            if best_num is None or kg_pct_candidates[0][0] > best_num:
                best_num = kg_pct_candidates[0][0]

        final_wt_selected_pct = ""
        if best_num is not None:
            final_wt_selected_pct = fmt_pct(best_num)

        # If nothing extracted at all, set final to empty
        if not (wt_llm_extracted or kg_values):
            final_wt_selected_pct = ""

        # Ensure final selected percent formatted
        def normalize_selected(s):
            if not s:
                return ""
            s2 = s.replace('%', '').replace(',', '.').strip()
            try:
                v = float(s2)
                v = abs(v)
                return fmt_pct(v)
            except:
                return ""

        final_wt_selected_pct = normalize_selected(final_wt_selected_pct)

        # Normalize hba selected
        def normalize_selected_hba(s):
            if not s:
                return ""
            s2 = s.replace('%', '').replace(',', '.').strip()
            try:
                v = float(s2)
                v = abs(v)
                return fmt_pct(v)
            except:
                return ""
        hba_selected = normalize_selected_hba(hba_selected)

        # Scores
        a1c_score = compute_a1c_score(hba_selected)
        weight_score = compute_weight_score(final_wt_selected_pct)

        # ---- LLM-based duration extraction (only when A1c or Weight score present) ----
        duration_str = duration_regex_str
        if _model is not None and (a1c_score or weight_score):
            dur_list, primary_dur = llm_extract_duration(_model, text_orig, drug_hint)
            # *** CHANGED BLOCK: use ONLY the LLM's single best duration ***
            if primary_dur:
                duration_str = primary_dur
            elif duration_regex_str:
                duration_str = duration_regex_str
            elif dur_list:
                # fallback: if LLM gives only a list, pick the first
                duration_str = dur_list[0]
        # -------------------------------------------------------------------------------

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
            'Weight selected %': final_wt_selected_pct,
            'Weight Score': weight_score,
            'duration': duration_str,  # duration now LLM-driven when scores present
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # keep row if hba or weight has extracted matches (regex), or if LLM extracted exist
    def has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(has_items))
    mask_keep = mask_hba | mask_wt
    out = out[mask_keep].reset_index(drop=True)

    # counts for UI
    out.attrs['counts'] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum()),
    )

    return out

# Run processing
out_df = process_df(model, df, col_name, drug_col)

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
# keep user columns order + these additions
# remove duplicates and keep order
seen = set()
new_cols = []
for c in cols:
    if c not in seen:
        new_cols.append(c)
        seen.add(c)
display_df = display_df[new_cols]

# Ensure 'duration' is the last column (move to end if present)
if 'duration' in display_df.columns:
    cols_no_duration = [c for c in display_df.columns if c != 'duration']
    cols_no_duration.append('duration')
    display_df = display_df[cols_no_duration]

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
def to_excel_bytes(df_out):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(display_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_llm_from_sentence.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
