# streamlit_hba1c_weight_llm_with_kg_fromto.py
# Full app with updated behavior:
# - For HbA1c "from X% to Y%" -> relative reduction ((X-Y)/X)*100 (existing behavior)
# - NEW: For weight "from x kg to y kg" -> relative reduction ((x - y) / y) * 100 (user-specified)
# - Baseline % (sidebar) is used only when no "from x kg to y kg" case exists.
# - Regex now recognizes kg-from->to and includes computed relative weight reduction in allowed set.
# - LLM is instructed to compute reductions, but the code will validate and force correct values.

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

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor (with kg from->to)", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (reads the sentence column)")

# -------------------- Sidebar options --------------------
st.sidebar.header("Options")
uploaded = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name = st.sidebar.text_input('Column with abstracts/text', value='abstract')
drug_col = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types, LLM_rejected)', value=False)
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
# kg from->to (new)
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
    # 1) percent from->to (already existing)
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

    # 1b) kg from->to (NEW: compute relative per user formula ((x - y)/y)*100)
    for m in re_fromto_kg.finditer(sent):
        a = parse_number(m.group(1))  # original x (kg)
        b = parse_number(m.group(2))  # new y (kg)
        reduction_val = None
        if (not math.isnan(a)) and (not math.isnan(b)) and (b != 0):
            try:
                # user-specified formula: ((x - y) / y) * 100
                reduction_val = ((a - b) / b) * 100.0
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

# -------------------- STRICT LLM RULES --------------------
LLM_RULES = """
You are an information-extraction assistant. Read the SENTENCE(S) provided and EXTRACT ONLY percentage CHANGES
related to the specified TARGET ("HbA1c" / "A1c" / "Body weight").

Return STRICT JSON only, with EXACT keys:
{
  "extracted": ["...%", "...%"],
  "selected_percent": "...%"
}

VERY IMPORTANT — Be conservative:
- Extract ONLY percentages that clearly refer to the TARGET and describe a CHANGE.
- DO NOT extract unrelated percentages (population %, p-values not representing change, baseline-only percentages).
- When seeing "from X% to Y%" for HbA1c/A1c, COMPUTE THE RELATIVE REDUCTION:
      ((X - Y) / X) * 100
- NEW: When seeing "from x kg to y kg" for Body weight, COMPUTE THE RELATIVE REDUCTION using the user-specified formula:
      ((x - y) / y) * 100
  Format with a '%' sign (3 decimals fine), include it in "extracted" and set as "selected_percent".
- For ranges like "1.2–1.5%", include both endpoints in "extracted" and prefer the HIGHER endpoint for "selected_percent".
- If multiple doses are present, prefer the change associated with the HIGHEST dose when choosing "selected_percent".
- Return JSON only. If nothing valid: {"extracted": [], "selected_percent": ""}.
"""

# -------------------- LLM extraction (returns percents, selected, kg_list, kg_fromto_computed_list) --------------------
def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_hint: str = "") -> Tuple[List[str], str, List[float], List[str]]:
    """
    Returns:
      - extracted_list_of_percents (strings like '1.23%'),
      - selected_percent (string like '1.23%'),
      - kg_list (list of floats found in LLM output or sentence),
      - kg_fromto_computed_list (list of percent strings computed from 'from x kg to y kg' found by regex)
    Behavior:
      - If 'from X% to Y%' (HbA1c) exists -> compute ((X-Y)/X)*100 and force it.
      - If 'from x kg to y kg' exists -> compute ((x - y) / y)*100 and force it for Body weight.
      - Baseline-based percent (sidebar) will be used **only** when no kg-from->to computed reduction exists.
    """
    # If model unavailable: still parse kg in sentence for fallback
    if model is None or not sentence.strip():
        kg_list = []
        kg_fromto_list = []
        for m in re_kg.finditer(sentence):
            num = parse_number(m.group(1))
            if not math.isnan(num):
                kg_list.append(num)
        # computed kg-from->to via regex
        for m in re_fromto_kg.finditer(sentence):
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            if not (math.isnan(a) or math.isnan(b)) and b != 0:
                try:
                    rel = ((a - b) / b) * 100.0
                    kg_fromto_list.append(fmt_pct(rel))
                except:
                    pass
        return [], "", kg_list, kg_fromto_list

    # 1) Compute authoritative reductions present in the sentence
    computed_hba1c_reductions = []
    computed_weight_fromto_reductions = []

    # HbA1c percent from->to (existing)
    if (target_label.lower().startswith('hba1c') or target_label.lower().startswith('a1c')):
        for m_ft in re_fromto.finditer(sentence):
            a = parse_number(m_ft.group(1)); b = parse_number(m_ft.group(2))
            if not (math.isnan(a) or math.isnan(b)) and a != 0:
                try:
                    rel = ((a - b) / a) * 100.0
                    computed_hba1c_reductions.append(fmt_pct(rel))
                except:
                    pass

    # Weight kg from->to (NEW per user formula ((x - y)/y)*100)
    for m_kg in re_fromto_kg.finditer(sentence):
        a = parse_number(m_kg.group(1)); b = parse_number(m_kg.group(2))
        if not (math.isnan(a) or math.isnan(b)) and b != 0:
            try:
                rel_w = ((a - b) / b) * 100.0
                computed_weight_fromto_reductions.append(fmt_pct(rel_w))
            except:
                pass

    # 2) Build prompt and call LLM
    prompt = (
        f"TARGET: {target_label}\n"
        + (f"DRUG_HINT: {drug_hint}\n" if drug_hint else "")
        + LLM_RULES + "\n"
        + "SENTENCE:\n" + sentence + "\n"
        + "Return JSON only.\n"
    )

    text = ""
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
    except Exception:
        text = ""

    # parse LLM JSON
    extracted_raw = []
    selected_raw = ""
    try:
        if text:
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e > s:
                data = json.loads(text[s:e+1])
                raw_ex = data.get("extracted") or []
                for x in raw_ex:
                    if isinstance(x, str):
                        extracted_raw.append(x.strip())
                selected_raw = (data.get("selected_percent") or "") or ""
    except Exception:
        extracted_raw = []
        selected_raw = ""

    # normalize percents and extract kg mentions from LLM tokens
    extracted = []
    kg_list = []
    for tok in extracted_raw:
        # detect kg inside LLM token (e.g., "2 kg")
        for m in re_kg.finditer(tok):
            num = parse_number(m.group(1))
            if not math.isnan(num):
                kg_list.append(num)
        xn = _norm_percent(tok)
        if re.search(r'[<>≥≤]', xn):
            continue
        if re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', xn):
            extracted.append(xn)

    # harvest kg from sentence as backup
    if not kg_list:
        for m in re_kg.finditer(sentence):
            num = parse_number(m.group(1))
            if not math.isnan(num):
                kg_list.append(num)

    # 3) FORCE computed reductions if present
    # HbA1c computed reductions (highest priority for HbA1c target)
    if computed_hba1c_reductions and (target_label.lower().startswith('hba1c') or target_label.lower().startswith('a1c')):
        comp = computed_hba1c_reductions[0]
        if comp not in extracted:
            extracted.insert(0, comp)
        selected = comp
        # final cleanup and return (we don't need kg info here)
        extracted = [e for e in extracted if re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', e)]
        return extracted, selected, kg_list, []

    # Weight kg-from->to computed reductions (force for Body weight target)
    if computed_weight_fromto_reductions and target_label.lower().startswith('body weight'.split()[0]) or (target_label.lower().startswith('body weight') or target_label.lower().startswith('weight')):
        # above condition ensures we handle weight-like labels; simplify check:
        if computed_weight_fromto_reductions:
            compw = computed_weight_fromto_reductions[0]
            if compw not in extracted:
                extracted.insert(0, compw)
            selected = compw
            # return and include the computed list so process_df knows not to use baseline
            return [e for e in extracted if re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', e)], selected, kg_list, computed_weight_fromto_reductions

    # 4) No forced computed reductions: validate LLM percents against regex allowed set
    allowed_set = _allowed_percents_from_regex(sentence, target_label)

    # Filter LLM-extracted percents: only keep those present in allowed_set (strict)
    if allowed_set:
        extracted = [e for e in extracted if e in allowed_set]
        selected = _norm_percent(selected_raw)
        if selected not in allowed_set:
            selected = ""
    else:
        selected = _norm_percent(selected_raw)

    # fallback proximity check if nothing matched but LLM provided percents
    if not extracted and extracted_raw:
        term_re = re_hba1c if (target_label.lower().startswith('hba1c') or target_label.lower().startswith('a1c')) else re_weight
        positions = [m.start() for m in term_re.finditer(sentence)]
        if positions:
            prox_allowed = []
            for tok in extracted_raw:
                xn = _norm_percent(tok)
                if not re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', xn):
                    continue
                ex_plain = xn.replace('%', '')
                found_near = False
                for p in positions:
                    left = max(0, p - 120)
                    right = min(len(sentence), p + 120)
                    window = sentence[left:right]
                    if re.search(re.escape(ex_plain) + r'\s*%', window):
                        found_near = True
                        break
                if found_near:
                    prox_allowed.append(xn)
            extracted = prox_allowed
            if selected and selected not in extracted:
                selected = ""

    # final selection: if still empty, return empty (conservative)
    if not extracted:
        return [], "", kg_list, []

    if not selected:
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

    extracted = [e for e in extracted if re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', e)]
    return extracted, selected, kg_list, []

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
                if not (math.isnan(a) or math.isnan(b)) and b != 0:
                    try:
                        rel = ((a - b) / b) * 100.0
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

        # DECISION LOGIC:
        # 1) If we have computed wt_kg_fromto_computed (from "from x kg to y kg"), use that as selected and do NOT use baseline.
        # 2) Else if kg_list present and baseline provided -> compute % = (kg_loss / baseline) * 100 and use that.
        # 3) Else keep wt_selected as returned/validated by LLM (or empty).

        # 1) weight from->to computed (priority)
        if wt_kg_fromto_computed:
            # prefer first computed (matches earlier behavior)
            chosen = wt_kg_fromto_computed[0]
            if chosen not in wt_llm_extracted:
                wt_llm_extracted.insert(0, chosen)
            wt_selected = chosen
        else:
            # 2) baseline-based percent only if no from->to present
            if wt_kg_list and baseline_kg and baseline_kg > 0:
                max_kg = max(wt_kg_list)
                pct_from_baseline = (max_kg / float(baseline_kg)) * 100.0
                wt_selected = fmt_pct(pct_from_baseline)
                if wt_selected not in wt_llm_extracted:
                    wt_llm_extracted.insert(0, wt_selected)
            # otherwise keep whatever wt_selected the LLM provided (possibly empty)

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
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types', 'LLM_rejected']:
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
    file_name='results_with_llm_and_weightkg_fromto.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
