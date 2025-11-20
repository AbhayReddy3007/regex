# streamlit_hba1c_weight_llm.py
# LLM is instructed to compute relative reductions for 'from X% to Y%' (HbA1c).
# The code validates the LLM output and forces the correct relative reduction if the LLM fails.

import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
API_KEY = ""   # <- replace with your real key
# =========================================================================

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

    Note: For tag_prefix == 'hba1c' we compute relative reduction ((X-Y)/X)*100 for from->to.
    """
    matches = []

    # 1) WHOLE-SENTENCE: "from X% to Y%"
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1))  # original X
        b = parse_number(m.group(2))  # new Y
        reduction_val = None
        if (not math.isnan(a)) and (not math.isnan(b)) and (a != 0):
            try:
                if tag_prefix == 'hba1c':
                    # relative reduction %
                    reduction_val = ((a - b) / a) * 100.0
                else:
                    # keep absolute percentage-point difference for non-HbA1c (legacy behavior)
                    reduction_val = (a - b)
            except Exception:
                reduction_val = None
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], reduction_val)

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
    """Return (matches, sentences_used) for sentences meeting the criterion.

    NOTE: we also include sentences that contain a 'from X% to Y%' pattern if the target
    term appears in the previous or next sentence (to capture 'from X to Y' phrasing that
    is split across sentences).
    """
    matches, sentences_used = [], []
    sents = split_sentences(text)
    for si, sent in enumerate(sents):
        keep = False
        if sentence_meets_criterion(sent, term_re):
            keep = True
        else:
            if re_fromto.search(sent):
                if term_re.search(sent):
                    keep = True
                else:
                    prev_has = (si > 0) and bool(term_re.search(sents[si-1]))
                    next_has = (si+1 < len(sents)) and bool(term_re.search(sents[si+1]))
                    if prev_has or next_has:
                        keep = True
        if not keep:
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

# -------------------- LLM RULES (now instructs LLM to compute relative reductions for from->to) --------------------
LLM_RULES = """
You are an information-extraction assistant. Your job is to read the SENTENCE(S) provided
and extract ONLY percentage CHANGES related to the specified TARGET:

- TARGET will be one of: "HbA1c", "A1c", or "Body weight".

Return STRICT JSON only (no explanations, no commentary), in this exact structure:

{
  "extracted": ["...%", "...%"],
  "selected_percent": "...%"
}

==============================================================
= CORE BEHAVIOR
==============================================================

1. **Extract only percentage CHANGE values.**
   - These include:
       • reductions
       • decreases
       • declines
       • drops
       • percentage-point reductions
       • ranges like “1.2–1.5%”
       • computed changes (explained below)
   - DO NOT return baseline percentages unless they represent a change.

2. **Normalize all extracted percentage values:**
   - They must end with a percent sign (%).
   - They must be numeric (no "<", ">", "≥", "≤", etc.).
   - For ranges like "1.2–1.5%", treat them as TWO values: 1.2% and 1.5%.

3. **Your output must contain:**
   - "extracted": an array of all valid extracted percentages (strings with %)
   - "selected_percent": the SINGLE best value for the target (string with %)

==============================================================
= SPECIAL RULE FOR 'from X% to Y%' (HbA1c / A1c only)
==============================================================

Whenever you see a phrase like:

- "from X% to Y%"
- "from X % to Y %"
- "from X% → Y%"

AND the TARGET is “HbA1c” or “A1c”:

### YOU MUST COMPUTE THE RELATIVE REDUCTION:

    relative_reduction = ((X - Y) / X) * 100

And then:

- Format it as a percentage (e.g., 13.415%).
- Include it in "extracted".
- Set it as the "selected_percent".

### DO NOT return the absolute difference (X - Y).  
Always return the RELATIVE reduction.

Examples:

Sentence: "HbA1c improved from 8.2% to 7.1%."
    X = 8.2, Y = 7.1
    relative = ((8.2 - 7.1) / 8.2) * 100 = 13.415%
    Output must include: "13.415%"
    selected_percent must be: "13.415%"

Sentence: "HbA1c levels declined from 7.80% to 6.90%."
    X = 7.80, Y = 6.90
    relative = ((7.8 - 6.9) / 7.8) * 100 = 11.538%
    selected_percent must be: "11.538%"

==============================================================
= RANGES
==============================================================

If the sentence lists ranges like:
- “1.2–1.5%”
- “1.4 to 1.8%”
- “1.4—1.8%”

You must:
1. Convert the range into its TWO numeric values.
2. Include BOTH in "extracted".
3. Prefer the HIGHER value when choosing "selected_percent", UNLESS the computed
   relative reduction (explained above) applies — in which case the computed value wins.

==============================================================
= MULTIPLE DOSES OR MULTIPLE CHANGES
==============================================================

If the sentence describes multiple doses (e.g., 0.5 mg, 1 mg, 2 mg), extract all changes.
Choose the change value corresponding to:

- the **highest dose** mentioned,
OR
- the explicitly stated primary finding,
OR
- the computed relative reduction (if present), which OVERRIDES all others.

==============================================================
= WHAT NOT TO RETURN
==============================================================

Do NOT output:
- Baseline percentages (e.g., “8.2%”) unless they are part of a change like “from X% to Y%”.
- Percentages related to unrelated outcomes.
- Percentages without change context.
- Thresholds (e.g., "<7%").

==============================================================
= OUTPUT FORMAT (JSON ONLY)
==============================================================

Return ONLY:

{
  "extracted": ["...%", "...%"],
  "selected_percent": "...%"
}

No text outside the JSON.
"""


def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_hint: str = ""):
    """
    Returns (extracted_list (strings like '1.23%'), selected_percent (string like '1.23%') )
    Behavior:
    - If sentence contains a from->to pattern for HbA1c, the LLM is instructed to compute relative reduction.
    - After model returns, the function validates: if a valid from->to exists and the LLM did not return
      the computed relative reduction, the code WILL compute and force the correct relative reduction to
      ensure accuracy.
    """
    if model is None or not sentence.strip():
        return [], ""

    # Build prompt
    prompt = (
        f"TARGET: {target_label}\n"
        + (f"DRUG_HINT: {drug_hint}\n" if drug_hint else "")
        + LLM_RULES + "\n"
        + "SENTENCE:\n" + sentence + "\n"
        + "Return JSON only.\n"
    )

    # Call model (if available)
    try:
        resp = model.generate_content(prompt) if model is not None else None
        text = (getattr(resp, "text", "") or "").strip() if resp is not None else ""
    except Exception:
        text = ""

    # parse JSON from model output if present
    extracted = []
    selected = ""
    try:
        if text:
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e > s:
                data = json.loads(text[s:e+1])
                raw_ex = data.get("extracted") or []
                for x in raw_ex:
                    if isinstance(x, str):
                        x_norm = _norm_percent(x)
                        if not re.search(r'[<>≥≤]', x_norm):
                            extracted.append(x_norm)
                selected = _norm_percent(data.get("selected_percent", "") or "")
    except Exception:
        extracted = []
        selected = ""

    # --- Validation & enforced-correctness for HbA1c from->to ---
    # If the SENTENCE contains a 'from X% to Y%' for HbA1c, compute the relative reduction expected.
    computed_reductions = []
    if (target_label.lower().startswith('hba1c') or target_label.lower().startswith('a1c')):
        for m_ft in re_fromto.finditer(sentence):
            a = parse_number(m_ft.group(1)); b = parse_number(m_ft.group(2))
            if not (math.isnan(a) or math.isnan(b)) and a != 0:
                try:
                    rel = ((a - b) / a) * 100.0
                    computed_reductions.append(fmt_pct(rel))
                except Exception:
                    pass

    # If LLM didn't return anything, keep as empty
    # But if computed_reductions exist, we strongly prefer the computed value:
    if computed_reductions:
        comp = computed_reductions[0]
        # If LLM already returned the same computed value, keep it.
        if comp in extracted and selected == comp:
            return extracted, selected
        # If LLM failed to return the computed value, we *force* it (but only after instructing LLM).
        # This guarantees correctness while still having LLM do the work when it follows instructions.
        # We insert comp at front of extracted and set selected to comp.
        if comp not in extracted:
            extracted.insert(0, comp)
        selected = comp
        # Return forced computed result
        return extracted, selected

    # --- No from->to computed case: postprocess LLM outputs (ranges -> max endpoint, dose-aware, final max) ---

    # normalize ranges in extracted to their highest endpoint
    range_re = re.compile(r'([+-]?\d+(?:[.,·]\d+)?)\s*(?:-|–|—|\sto\s|\s–\s)\s*([+-]?\d+(?:[.,·]\d+)?)\s*%?$', FLAGS)
    norm_extracted = []
    for x in extracted:
        if not isinstance(x, str):
            continue
        m = range_re.search(x)
        if m:
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            if not (math.isnan(a) or math.isnan(b)):
                norm_extracted.append(fmt_pct(max(a, b)))
            else:
                norm_extracted.append(x)
        else:
            norm_extracted.append(x)
    extracted = norm_extracted

    # If selected is not present or not normalized, set to normalized form
    selected = _norm_percent(selected)

    # If selected empty and extracted present, pick numeric max across extracted
    if (not selected) and extracted:
        max_val = None
        for ex in extracted:
            try:
                num = parse_number(ex.replace('%', ''))
                if math.isnan(num):
                    continue
                if max_val is None or num > max_val:
                    max_val = num
            except:
                continue
        if max_val is not None:
            selected = fmt_pct(max_val)

    # Final cleanup: only keep percent-formatted extracted
    extracted = [e for e in extracted if re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', e)]

    return extracted, selected

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

# -------------------- Processing function (note _model to avoid caching errors) --------------------
@st.cache_data
def process_df(_model, df_in: pd.DataFrame, text_col: str, drug_col_name: str):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, '')
        if not isinstance(text_orig, str):
            text_orig = '' if pd.isna(text_orig) else str(text_orig)

        # Run regex extraction to produce the 'sentence' columns (only sentences that meet the strict criteria)
        hba_matches, hba_sentences = extract_sentences(text_orig, re_hba1c, 'hba1c')
        wt_matches, wt_sentences   = extract_sentences(text_orig, re_weight, 'weight')

        # Format regex outputs
        def fmt_extracted(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                # present as percent (for HbA1c it's relative reduction, for weight it's kept as stored)
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
        hba_llm_extracted, hba_selected = ([], "")
        if _model is not None and sentence_str:
            hba_llm_extracted, hba_selected = llm_extract_from_sentence(_model, "HbA1c", sentence_str, drug_hint)

        # Weight LLM
        wt_llm_extracted, wt_selected = ([], "")
        if _model is not None and weight_sentence_str:
            wt_llm_extracted, wt_selected = llm_extract_from_sentence(_model, "Body weight", weight_sentence_str, drug_hint)

        # If LLM extracted is empty, we must ensure selected remains empty (explicit requirement)
        if not hba_llm_extracted:
            hba_selected = ""
        if not wt_llm_extracted:
            wt_selected = ""

        # Ensure selected% is formatted and positive
        def normalize_selected(s):
            if not s:
                return ""
            s2 = s.replace('%', '').replace(',', '.').strip()
            try:
                v = float(s2)
                v = abs(v)  # make positive on user's request
                return fmt_pct(v)
            except:
                return ""

        hba_selected = normalize_selected(hba_selected)
        wt_selected  = normalize_selected(wt_selected)

        # Scores
        a1c_score = compute_a1c_score(hba_selected)
        weight_score = compute_weight_score(wt_selected)

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

    # keep row if hba or weight has extracted matches (regex), or if LLM extracted exist (we assume we want rows with any data)
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

# Run processing (note we pass `model` in but process_df signature uses _model — avoids caching hash error)
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
display_df = display_df[cols]

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
