# streamlit_strict_hba1c_weight_llm.py
# Full app with robust LLM grounding and VERY DETAILED LLM RULES
# - Deterministic LLM calls (temperature=0)
# - Authoritative grounding: only allow tokens computed by regex or by explicit from->to
# - Weight: handles percent tokens, from x->y kg (compute ((x-y)/x)*100), and baseline fallback

import re
import math
import json
from io import BytesIO
from typing import List, Tuple

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
API_KEY = ""   # <- replace with your real key if you want LLM enabled
# =========================================================================

# Lazy Gemini import so the app still runs without the package
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor (strict LLM)", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini (strict, deterministic)")

# -------------------- Sidebar options --------------------
st.sidebar.header("Options")
uploaded = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name = st.sidebar.text_input('Column with abstracts/text', value='abstract')
drug_col = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (raw LLM output, rejected values)', value=False)
use_llm = st.sidebar.checkbox('Enable Gemini LLM (Gemini 2.0 Flash)', value=True)
baseline_weight_kg = st.sidebar.number_input(
    'Baseline weight for percent calc (kg)', value=70.0, min_value=1.0, step=0.1, format="%.1f"
)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
    st.stop()

# -------------------- Regex helpers --------------------
NUM = r'(?:[+-]?\d+(?:[.,·]\d+)?)'
PCT = rf'({NUM})\s*%'
DASH = r'(?:-|–|—)'

# percent from->to (for percents)
FROM_TO   = rf'from\s+({NUM})\s*%\s*(?:to|->|{DASH})\s*({NUM})\s*%'
# kg from->to
FROM_TO_KG = rf'from\s+({NUM})\s*(?:kg|kgs|kilogram|kilograms)\s*(?:to|->|{DASH})\s*({NUM})\s*(?:kg|kgs|kilogram|kilograms)?'
REDUCE_BY = rf'(?:reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\s*(?:by\s*)?({NUM})\s*%'
ABS_PP    = rf'(?:absolute\s+reduction\s+of|reduction\s+of)\s*({NUM})\s*%'
RANGE_PCT = rf'({NUM})\s*{DASH}\s*({NUM})\s*%'

FLAGS = re.IGNORECASE
re_pct       = re.compile(PCT, FLAGS)
re_fromto    = re.compile(FROM_TO, FLAGS)
re_fromto_kg = re.compile(FROM_TO_KG, FLAGS)
re_reduce_by = re.compile(REDUCE_BY, FLAGS)
re_abs_pp    = re.compile(ABS_PP, FLAGS)
re_range     = re.compile(RANGE_PCT, FLAGS)

re_hba1c  = re.compile(r'\bhb\s*a1c\b|\bhba1c\b|\ba1c\b', FLAGS)
re_weight = re.compile(r'\b(body\s*weight|weight|bw)\b', FLAGS)
re_reduction_cue = re.compile(
    r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b',
    FLAGS
)

# kg regex for capturing kg losses
re_kg = re.compile(r'([+-]?\d+(?:[.,·]\d+)?)\s*(?:kg|kgs|kilogram|kilograms)\b', FLAGS)

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

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

# Inclusive window by spaces (keeps nearby words)
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

# -------------------- Core extraction functions --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []
    # 1) percent from->to
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1))
        b = parse_number(m.group(2))
        reduction_val = None
        if (not math.isnan(a)) and (not math.isnan(b)) and (a != 0):
            try:
                if tag_prefix == 'hba1c':
                    reduction_val = ((a - b) / a) * 100.0
                else:
                    reduction_val = (a - b)
            except Exception:
                reduction_val = None
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], reduction_val)

    # 1b) kg from->to — compute relative reduction using ((x - y) / x) * 100
    for m in re_fromto_kg.finditer(sent):
        a = parse_number(m.group(1))  # original x (kg)
        b = parse_number(m.group(2))  # new y (kg)
        reduction_val = None
        if (not math.isnan(a)) and (not math.isnan(b)) and (a != 0):
            try:
                reduction_val = ((a - b) / a) * 100.0
            except:
                reduction_val = None
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_kg', [a, b], reduction_val)

    # 2) ±5-spaces window around target term
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

    # 3) weight fallback: previous % within 60 chars
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

    # dedupe & sort
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
    sents = split_sentences(text)
    for si, sent in enumerate(sents):
        keep = False
        if sentence_meets_criterion(sent, term_re):
            keep = True
        else:
            if re_fromto.search(sent) or re_fromto_kg.search(sent):
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

    seen, filtered = set(), []
    for mm in matches:
        key = (mm['sentence_index'], mm['span'])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(mm)
    filtered.sort(key=lambda x: (x['sentence_index'], x['span'][0]))
    return filtered, sentences_used

# -------------------- Core criterion (kg allowed) --------------------
def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    """
    Require:
      - the target term (matched by term_re) is present
      - a reduction cue is present (re_reduction_cue)
      - and EITHER a percent token (re_pct) OR the unit 'kg' appears in the sentence
    """
    if not isinstance(sent, str):
        return False
    has_term = bool(term_re.search(sent))
    has_cue  = bool(re_reduction_cue.search(sent))
    has_pct_or_kg = bool(re_pct.search(sent)) or bool(re_kg.search(sent)) or bool(re_fromto_kg.search(sent))
    return has_term and has_cue and has_pct_or_kg

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

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v

# -------------------- DETAILED LLM RULES (very explicit) --------------------
LLM_RULES = """
You are a strict, conservative information-extraction assistant. Read the provided SENTENCE(S) and EXTRACT ONLY percentage CHANGES that refer to the TARGET (one of: "HbA1c", "A1c", "Body weight").

OUTPUT (MANDATORY, JSON ONLY)
Return EXACTLY a single JSON object and nothing else, with these keys:
{
  "extracted": ["...%", "...%"],    // array of percent strings (each must exist in the sentence OR be computed from a present from->to)
  "selected_percent": "...%"        // single percent string chosen as the best candidate (or "" if none)
}

VERY IMPORTANT RULES (do not break)
1) ONLY include percentages that:
   - are exact percent tokens present in the SENTENCE (e.g., "12.3%") AND clearly tied to the TARGET, OR
   - are computed from an explicit "from X% to Y%" or "from x kg to y kg" pattern present in the SENTENCE (see formulas below).
   If a percent is not present and not computable, DO NOT include it.

2) When you see "from X% to Y%" and TARGET is HbA1c/A1c:
   - Compute relative reduction = ((X - Y) / X) * 100
   - Include the computed percent in "extracted" and set as "selected_percent".
   - Do not use (X - Y) as the selected percent.

3) When you see "from x kg to y kg" and TARGET is Body weight:
   - Compute relative reduction = ((x - y) / x) * 100
   - Include the computed percent in "extracted" and set as "selected_percent".

4) For ranges like "1.2–1.5%" or "1.2 to 1.5%":
   - Add both endpoints to "extracted" (e.g., ["1.2%","1.5%"]).
   - Prefer the HIGHER endpoint as "selected_percent" unless a computed from->to exists.

5) Proximity / context:
   - A percent token can be extracted only if:
     a) The sentence contains the TARGET term (or an immediately adjacent sentence ties the percent to the TARGET), AND
     b) The sentence contains a reduction cue (reduced/decreased/fell/lowered/from/declined/dropped), OR the percent is part of a from->to pattern.
   - If percent is clearly unrelated (p-values, percent of patients, CI bounds), DO NOT extract.

6) Dose handling:
   - If multiple doses appear, extract valid percent-changes and prefer the percent for the HIGHEST dose when choosing selected_percent.
   - If dose association cannot be established, choose the highest numeric percent.

7) Verification (critical):
   - For every percent you add to "extracted", you must either:
     a) locate the exact percent token in the SENTENCE (after normalization), OR
     b) compute it from an explicit from->to present in the sentence.
   - If you cannot demonstrate either, DO NOT extract.

8) Formatting:
   - Use '.' for decimals.
   - Strings must match regex: ^[+-]?\d+(?:\.\d+)?%$
   - No '<', '>', '≥', '≤', 'p-value' tokens, or extra text in percent strings.
   - Return positive values (absolute).

9) Ambiguity:
   - If not >= 90% confident that a percent follows these rules, return {"extracted": [], "selected_percent": ""}.

Examples (required outputs):
1) "HbA1c improved from 8.2% to 7.1%." -> {"extracted":["13.415%"], "selected_percent":"13.415%"}
2) "From baseline 100.0 kg to 91.5 kg, median weight decreased." -> {"extracted":["8.5%"], "selected_percent":"8.5%"}
3) "0.5 mg lowered HbA1c by 1.2-1.5% while 1.0 mg reduced 1.4-1.8%." -> {"extracted":["1.2%","1.5%","1.4%","1.8%"], "selected_percent":"1.8%"}
4) "Body weight decreased by 12.3% (p<0.001)." -> {"extracted":["12.3%"], "selected_percent":"12.3%"}

Negative examples (must return empty):
- "p=0.03, weight reduction significant." => {"extracted": [], "selected_percent": ""}
- "50% of patients achieved response." => {"extracted": [], "selected_percent": ""}
- "HbA1c baseline was 8.2%." => {"extracted": [], "selected_percent": ""}

FINAL: Return JSON only and nothing else.
"""

# -------------------- LLM extraction (strict, deterministic) --------------------
def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_hint: str = "") -> Tuple[List[str], str, List[float], List[str]]:
    """
    Returns:
      - extracted_list_of_percents (strings like '1.23%'),
      - selected_percent (string like '1.23%'),
      - kg_list (list of floats found in LLM output or sentence),
      - kg_fromto_computed_list (list of percent strings computed from 'from x kg to y kg' found by regex)
    Behavior:
      - Precompute authoritative set (regex-found percents + computed from->to reductions).
      - Pass ALLOWED_PERCENTS into prompt and set temperature=0.
      - Strictly accept only values from authoritative set or computed-from->to values.
    """
    if model is None or not sentence.strip():
        # still detect kg in sentence and kg-from->to computed values when model is not available
        kg_list = []
        kg_fromto_list = []
        for m in re_kg.finditer(sentence):
            num = parse_number(m.group(1))
            if not math.isnan(num):
                kg_list.append(num)
        for m in re_fromto_kg.finditer(sentence):
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            if not (math.isnan(a) or math.isnan(b)) and a != 0:
                try:
                    rel = ((a - b) / a) * 100.0
                    kg_fromto_list.append(fmt_pct(rel))
                except:
                    pass
        return [], "", kg_list, kg_fromto_list

    # Build authoritative sets
    authoritative = []

    # 1) computed reductions from explicit from->to in sentence (highest priority)
    computed = []
    # HbA1c percent from->to
    if target_label.lower().startswith('hba1c') or target_label.lower().startswith('a1c'):
        for m in re_fromto.finditer(sentence):
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            if not (math.isnan(a) or math.isnan(b)) and a != 0:
                try:
                    computed.append(fmt_pct(((a - b) / a) * 100.0))
                except:
                    pass
    # weight kg from->to
    for m in re_fromto_kg.finditer(sentence):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        if not (math.isnan(a) or math.isnan(b)) and a != 0:
            try:
                computed.append(fmt_pct(((a - b) / a) * 100.0))
            except:
                pass

    # 2) regex-found percents and ranges
    allowed_via_regex = sorted(list(_allowed_percents_from_regex(sentence, target_label)))
    # authoritative: computed first then regex tokens (computed wins)
    authoritative = list(dict.fromkeys(computed + allowed_via_regex))

    # Also harvest kg tokens directly from sentence (for baseline fallback)
    kg_list = []
    for m in re_kg.finditer(sentence):
        num = parse_number(m.group(1))
        if not math.isnan(num):
            kg_list.append(num)

    # Prepare prompt with grounding
    allowed_list_repr = ", ".join([f'"{p}"' for p in authoritative]) if authoritative else ""
    prompt_lines = []
    prompt_lines.append(f"TARGET: {target_label}")
    if drug_hint:
        prompt_lines.append(f"DRUG_HINT: {drug_hint}")
    prompt_lines.append(f"ALLOWED_PERCENTS: [{allowed_list_repr}]")
    prompt_lines.append("IMPORTANT: You MUST ONLY return JSON with keys 'extracted' and 'selected_percent'.")
    prompt_lines.append("You may only choose values from ALLOWED_PERCENTS or compute values from an explicit from->to pattern present in the SENTENCE.")
    prompt_lines.append(LLM_RULES)
    prompt_lines.append("SENTENCE:")
    prompt_lines.append(sentence)
    prompt = "\n".join(prompt_lines) + "\nReturn JSON only.\n"

    # Call model deterministically
    text = ""
    try:
        # Using generate_content with temperature=0.0 for deterministic output (SDK parameter)
        # If your SDK uses other param names adapt accordingly
        resp = model.generate_content(prompt, temperature=0.0)
        text = (getattr(resp, "text", "") or "").strip()
    except Exception:
        # model call failed -> return nothing but still include kg list
        return [], "", kg_list, computed

    # Parse JSON returned by LLM
    extracted_raw = []
    selected_raw = ""
    try:
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e <= s:
            return [], "", kg_list, computed
        data = json.loads(text[s:e+1])
        raw_ex = data.get("extracted") or []
        for x in raw_ex:
            if isinstance(x, str):
                extracted_raw.append(x.strip())
        selected_raw = (data.get("selected_percent") or "") or ""
    except Exception:
        # parsing failed
        return [], "", kg_list, computed

    # Normalize and validate LLM outputs against authoritative set
    extracted = []
    validated_rejected = []  # for debug
    for tok in extracted_raw:
        xn = _norm_percent(tok).replace(',', '.')
        if not re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', xn):
            validated_rejected.append(tok)
            continue
        # make absolute and format
        try:
            val = abs(parse_number(xn.replace('%', '')))
            xn = fmt_pct(val)
        except:
            validated_rejected.append(tok)
            continue
        if xn in authoritative:
            extracted.append(xn)
        else:
            # not in authoritative -> reject
            validated_rejected.append(tok)

    sel = _norm_percent(selected_raw).replace(',', '.')
    if sel and re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', sel):
        try:
            sel_val = abs(parse_number(sel.replace('%', '')))
            sel = fmt_pct(sel_val)
        except:
            sel = ""
    else:
        sel = ""

    # If computed reductions exist, force them (highest priority)
    if computed:
        comp = computed[0]
        if comp not in extracted:
            extracted.insert(0, comp)
        sel = comp

    # If selected provided by LLM but not authoritative, reject selection
    if sel and (sel not in authoritative):
        # reject; prefer authoritative first element if available
        if authoritative:
            sel = authoritative[0]
            if sel not in extracted:
                extracted.insert(0, sel)
        else:
            sel = ""

    # Final cleanup: ensure unique extracted and valid pattern
    uniq_extracted = []
    for e in extracted:
        if e not in uniq_extracted:
            uniq_extracted.append(e)
    extracted = uniq_extracted

    # If nothing valid extracted, return empty (conservative)
    if not extracted:
        return [], "", kg_list, computed

    return extracted, sel, kg_list, computed

# Helper: build allowed set from regex (used in validation)
def _allowed_percents_from_regex(sentence: str, target_label: str) -> set:
    if target_label.lower().startswith('hba1c') or target_label.lower().startswith('a1c'):
        term_re, tag = re_hba1c, 'hba1c'
    else:
        term_re, tag = re_weight, 'weight'

    matches, sents = extract_sentences(sentence, term_re, tag)
    allowed = set()
    for m in matches:
        t = (m.get('type') or '').lower()
        if 'from-to' in t:
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                allowed.add(fmt_pct(rp))
            vals = m.get('values') or []
            for v in vals:
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        allowed.add(fmt_pct(v))
                except:
                    pass
        elif 'range' in t:
            vals = m.get('values') or []
            if len(vals) >= 2:
                a, b = vals[0], vals[1]
                if not (math.isnan(a) or math.isnan(b)):
                    allowed.add(fmt_pct(a))
                    allowed.add(fmt_pct(b))
                    allowed.add(fmt_pct(max(a, b)))
        else:
            vals = m.get('values') or []
            for v in vals:
                if v is None:
                    continue
                if isinstance(v, (int, float)):
                    if not math.isnan(v):
                        allowed.add(fmt_pct(v))
                else:
                    try:
                        num = parse_number(v)
                        if not math.isnan(num):
                            allowed.add(fmt_pct(num))
                    except:
                        pass
    return allowed

# -------------------- Scoring helpers --------------------
def compute_a1c_score(selected_pct_str: str):
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
def process_df(_model, df_in: pd.DataFrame, text_col: str, drug_col_name: str, baseline_kg: float):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, '')
        if not isinstance(text_orig, str):
            text_orig = '' if pd.isna(text_orig) else str(text_orig)

        # regex extraction: sentences & matches
        hba_matches, hba_sentences = extract_sentences(text_orig, re_hba1c, 'hba1c')
        wt_matches, wt_sentences   = extract_sentences(text_orig, re_weight, 'weight')

        def fmt_extracted(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        hba_regex_vals = [fmt_extracted(m) for m in hba_matches]
        wt_regex_vals  = [fmt_extracted(m) for m in wt_matches]

        sentence_str = ' | '.join(hba_sentences) if hba_sentences else ''
        weight_sentence_str = ' | '.join(wt_sentences) if wt_sentences else ''

        drug_hint = ""
        if drug_col_name and drug_col_name in df_in.columns:
            drug_hint = str(row.get(drug_col_name, '') or "")

        # HbA1c LLM extraction (returns kg lists too, but ignored)
        hba_llm_extracted, hba_selected, _, _ = ([], "", [], [])
        if _model is not None and sentence_str:
            hba_llm_extracted, hba_selected, _, _ = llm_extract_from_sentence(_model, "HbA1c", sentence_str, drug_hint)

        # Weight LLM extraction -> returns percents, selected%, kg_list, kg_fromto_computed_list
        wt_llm_extracted, wt_selected, wt_kg_list, wt_kg_fromto_computed = ([], "", [], [])
        if _model is not None and weight_sentence_str:
            wt_llm_extracted, wt_selected, wt_kg_list, wt_kg_fromto_computed = llm_extract_from_sentence(_model, "Body weight", weight_sentence_str, drug_hint)
        else:
            # even without model, detect kg in sentence and kg-from->to computed reductions
            for mkg in re_kg.finditer(weight_sentence_str):
                num = parse_number(mkg.group(1))
                if not math.isnan(num):
                    wt_kg_list.append(num)
            for m in re_fromto_kg.finditer(weight_sentence_str):
                a = parse_number(m.group(1)); b = parse_number(m.group(2))
                if not (math.isnan(a) or math.isnan(b)) and a != 0:
                    try:
                        rel = ((a - b) / a) * 100.0
                        wt_kg_fromto_computed.append(fmt_pct(rel))
                    except:
                        pass

        # If LLM extracted is empty, ensure selected remains empty unless we have kg_fromto
        if not hba_llm_extracted:
            hba_selected = ""

        # Build 'weight KG' strings (from kg_list)
        weight_kg_strs = []
        for kgv in wt_kg_list:
            try:
                kgf = float(kgv)
                s = f"{kgf:.3f} kg"
                s = s.replace('.000 kg', ' kg').replace('.00 kg', ' kg').replace('.0 kg', ' kg')
                weight_kg_strs.append(s)
            except:
                pass

        # ---------------- DECISION LOGIC ----------------
        # Priority:
        # 1) If sentence contains explicit percent tokens for weight -> use percent (prefer LLM selected; else regex max).
        # 2) Else if no percent tokens:
        #    a) If kg-from->to computed exists -> use that computed percent (forced).
        #    b) Else if kg_list exists and baseline provided -> compute percent from baseline and use that.
        #    c) Else leave wt_selected as-is.

        sentence_has_pct = bool(re_pct.search(weight_sentence_str))

        if sentence_has_pct:
            # Prefer any LLM-selected percent (validated earlier)
            if wt_selected:
                pass
            else:
                # pick numeric max percent found in the sentence by regex
                pct_vals = []
                for m in re_pct.finditer(weight_sentence_str):
                    v = parse_number(m.group(1))
                    if not math.isnan(v):
                        pct_vals.append(v)
                if pct_vals:
                    max_pct = max(pct_vals)
                    wt_selected = fmt_pct(max_pct)
                    if wt_selected not in wt_llm_extracted:
                        wt_llm_extracted.insert(0, wt_selected)
            # do NOT use kg-from->to or baseline in this branch
        else:
            # no percent token in sentence -> try kg-from->to first (if present), else baseline-based kg -> percent
            if wt_kg_fromto_computed:
                chosen = wt_kg_fromto_computed[0]
                if chosen not in wt_llm_extracted:
                    wt_llm_extracted.insert(0, chosen)
                wt_selected = chosen
            else:
                if wt_kg_list and baseline_kg and baseline_kg > 0:
                    max_kg = max(wt_kg_list)
                    pct_from_baseline = (max_kg / float(baseline_kg)) * 100.0
                    wt_selected = fmt_pct(pct_from_baseline)
                    if wt_selected not in wt_llm_extracted:
                        wt_llm_extracted.insert(0, wt_selected)
                else:
                    pass

        # Ensure selected% formatted & positive
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

        hba_selected = normalize_selected(hba_selected)
        wt_selected = normalize_selected(wt_selected)

        # Scores (recompute weight score after possible override)
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
            'weight KG': weight_kg_strs,
            'Weight selected %': wt_selected,
            'Weight Score': weight_score,
        })
        if show_debug:
            # include raw debug info for troubleshooting
            new.update({
                'debug_kg_list': wt_kg_list,
                'debug_kg_fromto_computed': wt_kg_fromto_computed,
            })
        rows.append(new)

    out = pd.DataFrame(rows)

    def has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(has_items))
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

# -------------------- Read uploaded file --------------------
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

# -------------------- Run processing --------------------
out_df = process_df(model, df, col_name, drug_col, baseline_weight_kg)

# -------------------- Reorder columns: place LLM columns BESIDE regex columns and insert weight KG --------------------
def insert_after(cols, after, names):
    if after not in cols:
        cols.extend([n for n in names if n not in cols])
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
cols = insert_after(cols, "weight_extracted_matches", ["Weight LLM extracted", "weight KG", "Weight selected %", "Weight Score"])
display_df = display_df[[c for c in cols if c in display_df.columns]]

# -------------------- Show results --------------------
st.write("### Results (first 200 rows shown)")
# drop internal debug columns by default
if not show_debug:
    for c in ['debug_kg_list', 'debug_kg_fromto_computed']:
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
    file_name='results_with_strict_llm.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
