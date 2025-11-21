# streamlit_hba1c_weight_strict_debug_fallback_llm_fixed_full.py
# Full Streamlit app — copy-paste ready with fixes:
# - robust _norm_percent
# - improved configure_gemini + call_model_deterministic with better error messages
# - quick LLM test UI
# - behavior otherwise follows your previous strict+debug+fallback design

import re
import math
import json
from io import BytesIO
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
API_KEY = ""   # <- replace with your real key if you want LLM enabled
# =========================================================================

# Lazy Gemini import flag; we'll try to import inside configure_gemini
GENAI_AVAILABLE = False
genai = None  # will be assigned if import succeeds

st.set_page_config(page_title="HbA1c & Weight % Extractor (strict + debug + fallback LLM)", layout="wide")
st.title("HbA1c / A1c + Body Weight — strict + deterministic LLM extraction with debug & fallback")

# -------------------- Sidebar --------------------
st.sidebar.header("Options")
uploaded = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name = st.sidebar.text_input('Column with abstracts/text', value='abstract')
drug_col = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (raw LLM output, rejected values)', value=True)
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

FROM_TO   = rf'from\s+({NUM})\s*%\s*(?:to|->|{DASH})\s*({NUM})\s*%'
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

def _norm_percent(v: str) -> str:
    """
    Normalize a percent-like token into a canonical 'N%' string.
    Accepts numbers like '1.2', '1,2%', '1.2%' and returns '1.2%'.
    Leaves empty/invalid inputs as ''.
    This version is defensive: accepts floats/ints also, strips spaces and percent signs,
    accepts leading +/-, and returns absolute value formatted with fmt_pct.
    """
    if v is None:
        return ""
    # if it's already numeric
    if isinstance(v, (int, float)):
        try:
            return fmt_pct(abs(float(v)))
        except:
            return ""
    s = str(v).strip()
    if s == "":
        return ""
    # remove percent sign, spaces, thousands separators
    s = s.replace('%', '').replace(',', '.').replace('·', '.').replace(' ', '')
    # allow leading plus/minus, digits and optional decimal
    if not re.match(r'^[+-]?\d+(?:\.\d+)?$', s):
        return ""
    try:
        num = float(s)
    except:
        return ""
    return fmt_pct(abs(num))

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

# -------------------- Core extraction --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
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

    for m in re_fromto_kg.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        reduction_val = None
        if (not math.isnan(a)) and (not math.isnan(b)) and (a != 0):
            try:
                reduction_val = ((a - b) / a) * 100.0  # user-requested formula
            except:
                reduction_val = None
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_kg', [a, b], reduction_val)

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

def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    if not isinstance(sent, str):
        return False
    has_term = bool(term_re.search(sent))
    has_cue  = bool(re_reduction_cue.search(sent))
    has_pct_or_kg = bool(re_pct.search(sent)) or bool(re_kg.search(sent)) or bool(re_fromto_kg.search(sent))
    return has_term and has_cue and has_pct_or_kg

# -------------------- Robust Gemini configuration & deterministic call wrapper ----------
def configure_gemini(api_key: str) -> Tuple[Optional[object], str]:
    """
    Try to configure google.generativeai and return (model_handle_or_module_or_None, error_string).
    This version records full exceptions to help debug SDK mismatch/API key issues.
    """
    global genai, GENAI_AVAILABLE
    if not api_key:
        return None, "API_KEY empty"
    try:
        import google.generativeai as _genai
        genai = _genai
        GENAI_AVAILABLE = True
    except Exception as e:
        return None, f"Failed to import google.generativeai: {repr(e)}"
    # configure (may raise)
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        configure_err = f"Warning during genai.configure: {repr(e)}"
        try:
            model_obj = getattr(genai, "GenerativeModel", None)
            if model_obj:
                try:
                    return genai.GenerativeModel("gemini-2.0-flash"), configure_err
                except Exception as e2:
                    return genai, configure_err + f" ; Failed to init GenerativeModel: {repr(e2)}"
            else:
                return genai, configure_err
        except Exception as e2:
            return genai, configure_err + f" ; fallback error: {repr(e2)}"
    # if configure succeeded, try to return a model object if possible
    try:
        model_obj = genai.GenerativeModel("gemini-2.0-flash")
        return model_obj, ""
    except Exception as e:
        return genai, f"Configured but GenerativeModel() failed: {repr(e)}"

# call once
_model_handle, configure_error = configure_gemini(API_KEY if use_llm else "")
if configure_error:
    st.sidebar.warning(f"Gemini configure note: {configure_error}")

def call_model_deterministic(model_handle, prompt: str, timeout: int = 20) -> Tuple[str, str]:
    """
    Try multiple supported SDK call styles deterministically. Returns (text, error_str).
    """
    last_err = None
    # try model object style (newer SDKs)
    try:
        if model_handle is not None and hasattr(model_handle, "generate_content"):
            resp = model_handle.generate_content(prompt, temperature=0.0)
            text = getattr(resp, "text", None) or getattr(resp, "content", None) or ""
            if text:
                return text.strip(), ""
    except Exception as e:
        last_err = f"model_handle.generate_content failed: {repr(e)}"

    # try module-level helpers
    if 'genai' in globals() and genai is not None:
        try:
            if hasattr(genai, "generate_text"):
                resp = genai.generate_text(model="gemini-2.0-flash", prompt=prompt, temperature=0.0)
                if hasattr(resp, "text"):
                    text = getattr(resp, "text") or ""
                elif isinstance(resp, dict):
                    text = resp.get("candidates", [{}])[0].get("content", "") or resp.get("output", "")
                else:
                    text = str(resp)
                if text:
                    return text.strip(), ""
            if hasattr(genai, "models") and hasattr(genai.models, "generate"):
                resp = genai.models.generate(model="gemini-2.0-flash", prompt=prompt, temperature=0.0)
                if isinstance(resp, dict):
                    text = resp.get("candidates", [{}])[0].get("content", "") or resp.get("output", "") or resp.get("text", "")
                else:
                    text = getattr(resp, "text", None) or str(resp)
                if text:
                    return text.strip(), ""
            if hasattr(genai, "generate"):
                resp = genai.generate(model="gemini-2.0-flash", prompt=prompt, temperature=0.0)
                text = getattr(resp, "text", None) or (resp.get("candidates", [{}])[0].get("content", "") if isinstance(resp, dict) else None) or str(resp)
                if text:
                    return text.strip(), ""
        except Exception as e:
            return "", f"genai call error: {repr(e)}"

    if last_err:
        return "", last_err
    return "", "No supported model call succeeded; check SDK version / API key / network."

# -------------------- Strict LLM RULES (detailed) --------------------
LLM_RULES = """
You are a strict, conservative information-extraction assistant. Read the provided SENTENCE(S) and EXTRACT ONLY percentage CHANGES that refer to the TARGET (one of: "HbA1c", "A1c", "Body weight").

OUTPUT (MANDATORY, JSON ONLY)
Return EXACTLY a single JSON object and nothing else, with keys:
{
  "extracted": ["...%", "...%"],    // percent strings present in sentence or computed from from->to
  "selected_percent": "...%",        // single percent string chosen as the best candidate (or "" if none)
  "preference": "percent" | "kg"     // OPTIONAL — state which measure you prefer (percent or kg-derived).
}

VERY IMPORTANT RULES (do not break)
- ONLY include percentages that:
  * are exact percent tokens present in the SENTENCE AND clearly tied to TARGET, OR
  * are computed from an explicit "from X% to Y%" or "from x kg to y kg" present in the SENTENCE.
  If neither, DO NOT include.

- For "from X% to Y%" (HbA1c): compute ((X - Y) / X) * 100 and include it.
- For "from x kg to y kg" (Body weight): compute ((x - y) / x) * 100 and include it.
- For ranges like "1.2–1.5%": include both endpoints; prefer the HIGHER endpoint as selected.
- A percent may be extracted only if sentence contains TARGET (or adjacent sentence ties it) AND a reduction cue, OR it's part of from->to.
- If multiple doses reported, prefer highest-dose-associated change for selection; if unclear, choose highest numeric percent.
- Each percent must be traceable (exact token or computed); otherwise DO NOT extract.
- Formatting: use '.' decimal, match regex ^[+-]?\d+(?:\.\d+)?%$, positive absolute values.
- If not confident, return {"extracted": [], "selected_percent": "", "preference": ""}

Return JSON only.
"""

# -------------------- LLM extraction with debug, deterministic wrapper, and fallback support --------------------
def llm_extract_from_sentence(model_handle, target_label: str, sentence: str, drug_hint: str = "") -> Tuple[List[str], str, List[float], List[str], str, str]:
    """
    Returns:
      - extracted (list of percent strings)
      - selected_percent (single percent string)
      - kg_list (list of kg numbers found in sentence)
      - computed_fromto_list (computed percent strings from from->to patterns)
      - preference ('percent'|'kg'|'')
      - debug_json (stringified JSON containing raw_text, model_error, configure_error)
    """
    debug_info = {"raw_text": "", "model_error": "", "configure_error": configure_error or ""}

    # If model unavailable, still harvest kg tokens and computed from->to
    if model_handle is None:
        kg_list = []
        for m in re_kg.finditer(sentence):
            num = parse_number(m.group(1))
            if not math.isnan(num):
                kg_list.append(num)
        kg_fromto_list = []
        for m in re_fromto_kg.finditer(sentence):
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            if not (math.isnan(a) or math.isnan(b)) and a != 0:
                try:
                    kg_fromto_list.append(fmt_pct(((a - b) / a) * 100.0))
                except:
                    pass
        debug_info["model_error"] = "Model disabled or not configured."
        return [], "", kg_list, kg_fromto_list, "", json.dumps(debug_info)

    # compute authoritative computed reductions from explicit from->to patterns
    computed = []
    if target_label.lower().startswith('hba1c') or target_label.lower().startswith('a1c'):
        for m in re_fromto.finditer(sentence):
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            if not (math.isnan(a) or math.isnan(b)) and a != 0:
                try:
                    computed.append(fmt_pct(((a - b) / a) * 100.0))
                except:
                    pass
    for m in re_fromto_kg.finditer(sentence):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        if not (math.isnan(a) or math.isnan(b)) and a != 0:
            try:
                computed.append(fmt_pct(((a - b) / a) * 100.0))
            except:
                pass

    allowed_via_regex = sorted(list(_allowed_percents_from_regex(sentence, target_label)))

    # contextual filter for weight target to avoid A1c bleed
    if target_label.lower().startswith('body weight') or target_label.lower().startswith('weight'):
        weight_pos = [m.start() for m in re_weight.finditer(sentence)]
        a1c_pos = [m.start() for m in re_hba1c.finditer(sentence)]
        def closest_distance(idx_list, pos):
            if not idx_list:
                return float('inf')
            return min(abs(pos - p) for p in idx_list)
        filtered_allowed = set()
        for m in re_pct.finditer(sentence):
            raw = fmt_pct(parse_number(m.group(1)))
            start_pos = m.start()
            dist_w = closest_distance(weight_pos, start_pos)
            dist_a = closest_distance(a1c_pos, start_pos)
            if dist_w <= dist_a or not a1c_pos:
                filtered_allowed.add(raw)
        allowed_via_regex = [p for p in allowed_via_regex if p in filtered_allowed or p in computed]

    authoritative = list(dict.fromkeys(computed + allowed_via_regex))

    # harvest kg tokens
    kg_list = [parse_number(m.group(1)) for m in re_kg.finditer(sentence) if not math.isnan(parse_number(m.group(1)))]

    # detect mean-decrease proximity hint
    mean_phrases = re.search(r'\bmean\s+(?:decreas|reduc|change|decline|improv|fall)', sentence, re.IGNORECASE)
    mean_nearest_hint = ""
    if mean_phrases:
        ph_start = mean_phrases.start()
        nearest_pct = None; nearest_pct_dist = float('inf')
        for m in re_pct.finditer(sentence):
            dist = abs(m.start() - ph_start)
            if dist < nearest_pct_dist:
                nearest_pct_dist = dist
                nearest_pct = fmt_pct(parse_number(m.group(1)))
        nearest_kg = None; nearest_kg_dist = float('inf')
        for m in re_kg.finditer(sentence):
            dist = abs(m.start() - ph_start)
            if dist < nearest_kg_dist:
                nearest_kg_dist = dist
                nearest_kg = parse_number(m.group(1))
        if nearest_pct and (nearest_pct_dist <= nearest_kg_dist):
            mean_nearest_hint = "percent"
        elif nearest_kg:
            mean_nearest_hint = "kg"

    # build prompt with grounding (ALLOWED_PERCENTS)
    allowed_list_repr = ", ".join([f'"{p}"' for p in authoritative]) if authoritative else ""
    prompt_lines = [
        f"TARGET: {target_label}"
    ]
    if drug_hint:
        prompt_lines.append(f"DRUG_HINT: {drug_hint}")
    prompt_lines.append(f"ALLOWED_PERCENTS: [{allowed_list_repr}]")
    prompt_lines.append("IMPORTANT: You MUST ONLY return JSON with keys 'extracted','selected_percent' and optionally 'preference' ('percent' or 'kg').")
    if mean_phrases:
        prompt_lines.append("NOTE: Sentence contains 'mean' phrasing; prefer value nearest that phrase and state preference.")
    prompt_lines.append(LLM_RULES)
    prompt_lines.append("SENTENCE:")
    prompt_lines.append(sentence)
    prompt_lines.append("Return JSON only. Example: {\"extracted\": [\"1.2%\",\"1.5%\"], \"selected_percent\": \"1.5%\", \"preference\":\"percent\"}")
    prompt = "\n".join(prompt_lines) + "\n"

    # call model deterministically using wrapper
    text, model_error = call_model_deterministic(model_handle, prompt)
    debug_info["raw_text"] = text or ""
    debug_info["model_error"] = model_error or ""
    debug_info["configure_error"] = configure_error or ""

    if not text:
        # fallback: return conservative regex/computed results and debug info
        return [], "", kg_list, computed, "", json.dumps(debug_info)

    # parse JSON from model text
    try:
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e <= s:
            return [], "", kg_list, computed, "", json.dumps(debug_info)
        data = json.loads(text[s:e+1])
    except Exception:
        return [], "", kg_list, computed, "", json.dumps(debug_info)

    raw_ex = data.get("extracted", []) or []
    raw_sel = data.get("selected_percent", "") or ""
    raw_pref = (data.get("preference", "") or "").lower()

    # normalize & validate extracted tokens vs authoritative
    extracted = []
    for tok in raw_ex:
        if not isinstance(tok, str):
            continue
        xn = _norm_percent(tok).replace(',', '.')
        if not re.match(r'^[+-]?\d+(?:\.\d+)?%$', xn):
            continue
        try:
            xn = fmt_pct(abs(parse_number(xn.replace('%', ''))))
        except:
            continue
        if xn in authoritative:
            extracted.append(xn)

    sel = _norm_percent(raw_sel).replace(',', '.')
    if sel and re.match(r'^[+-]?\d+(?:\.\d+)?%$', sel):
        try:
            sel = fmt_pct(abs(parse_number(sel.replace('%', ''))))
        except:
            sel = ""
    else:
        sel = ""

    # computed reductions override authoritative picks
    if computed:
        comp = computed[0]
        if comp not in extracted:
            extracted.insert(0, comp)
        sel = comp
        pref = "percent"
        return extracted, sel, kg_list, computed, pref, json.dumps(debug_info)

    # preference handling
    pref = ""
    if raw_pref in ("percent", "kg"):
        pref = raw_pref
    elif mean_nearest_hint:
        pref = mean_nearest_hint

    if sel and (sel not in authoritative):
        sel = ""

    if not extracted and authoritative:
        extracted = authoritative.copy()

    if not extracted:
        return [], "", kg_list, computed, pref, json.dumps(debug_info)

    return extracted, sel, kg_list, computed, pref, json.dumps(debug_info)

# -------------------- Helper: allowed percents from regex --------------------
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
def process_df(_model_handle, df_in: pd.DataFrame, text_col: str, drug_col_name: str, baseline_kg: float):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, '')
        if not isinstance(text_orig, str):
            text_orig = '' if pd.isna(text_orig) else str(text_orig)

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

        # HbA1c LLM extraction
        hba_llm_extracted, hba_selected, _, _, _, hba_debug = ([], "", [], [], "", "")
        if use_llm and _model_handle is not None and sentence_str:
            hba_llm_extracted, hba_selected, _, _, _, hba_debug = llm_extract_from_sentence(_model_handle, "HbA1c", sentence_str, drug_hint)
        else:
            hba_debug = json.dumps({"model_error": "Model disabled or not configured."})

        # Automatic fallback for HbA1c: if debug shows error or raw_text empty or no extracted values -> discard LLM outputs
        try:
            dbg_h = json.loads(hba_debug) if isinstance(hba_debug, str) and hba_debug else {}
        except:
            dbg_h = {}
        if (not hba_llm_extracted) or dbg_h.get("model_error") or not dbg_h.get("raw_text"):
            hba_llm_extracted = []
            hba_selected = ""

        # Weight LLM extraction (returns preference and debug)
        wt_llm_extracted, wt_selected, wt_kg_list, wt_kg_fromto_computed, wt_pref, wt_debug = ([], "", [], [], "", "")
        if use_llm and _model_handle is not None and weight_sentence_str:
            wt_llm_extracted, wt_selected, wt_kg_list, wt_kg_fromto_computed, wt_pref, wt_debug = llm_extract_from_sentence(_model_handle, "Body weight", weight_sentence_str, drug_hint)
        else:
            # parse kg tokens and computed kg-from->to when model not available
            for mkg in re_kg.finditer(weight_sentence_str):
                num = parse_number(mkg.group(1))
                if not math.isnan(num):
                    wt_kg_list.append(num)
            for m in re_fromto_kg.finditer(weight_sentence_str):
                a = parse_number(m.group(1)); b = parse_number(m.group(2))
                if not (math.isnan(a) or math.isnan(b)) and a != 0:
                    try:
                        wt_kg_fromto_computed.append(fmt_pct(((a - b) / a) * 100.0))
                    except:
                        pass
            wt_debug = json.dumps({"model_error": "Model disabled or not configured."})

        # Automatic fallback for Weight: if debug shows error or raw_text empty or no extracted values -> discard LLM percent outputs
        try:
            dbg_w = json.loads(wt_debug) if isinstance(wt_debug, str) and wt_debug else {}
        except:
            dbg_w = {}
        if (not wt_llm_extracted) or dbg_w.get("model_error") or not dbg_w.get("raw_text"):
            # Discard LLM percent outputs but keep kg lists & computed kg-from->to (regex-based)
            wt_llm_extracted = []
            wt_selected = ""
            wt_pref = ""

        # If LLM extracted is empty, ensure selected remains empty unless we have kg_fromto
        if not hba_llm_extracted:
            hba_selected = ""

        # Build 'weight KG' strings
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
        # Determine if sentence contains explicit weight percent tokens tied to weight
        sentence_has_pct = False
        for m in re_pct.finditer(weight_sentence_str):
            start_pos = m.start()
            weight_pos = [p.start() for p in re_weight.finditer(weight_sentence_str)]
            a1c_pos = [p.start() for p in re_hba1c.finditer(weight_sentence_str)]
            def closest_distance(idx_list, pos):
                if not idx_list:
                    return float('inf')
                return min(abs(pos - p) for p in idx_list)
            if closest_distance(weight_pos, start_pos) <= closest_distance(a1c_pos, start_pos):
                sentence_has_pct = True
                break

        chosen_from = ""
        # 1) If percent tokens present tied to weight -> prefer percent
        if sentence_has_pct:
            if wt_selected:
                chosen_from = "percent"
            elif wt_llm_extracted:
                chosen_from = "percent"
            else:
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
                    chosen_from = "percent"

        # 2) If no percent tokens tied to weight -> consider preference / kg-from->to / baseline
        if not chosen_from:
            if wt_pref == "kg" and (wt_kg_fromto_computed or wt_kg_list):
                chosen_from = "kg"
            elif wt_pref == "percent" and (wt_llm_extracted or sentence_has_pct):
                chosen_from = "percent"
            else:
                if wt_kg_fromto_computed:
                    chosen_from = "kg"
                elif wt_kg_list and not sentence_has_pct:
                    chosen_from = "kg"
                elif wt_llm_extracted:
                    chosen_from = "percent"
                else:
                    chosen_from = ""

        # Apply chosen result
        if chosen_from == "kg":
            if wt_kg_fromto_computed:
                wt_selected = wt_kg_fromto_computed[0]
                if wt_selected not in wt_llm_extracted:
                    wt_llm_extracted.insert(0, wt_selected)
            elif wt_kg_list and baseline_kg and baseline_kg > 0:
                max_kg = max(wt_kg_list)
                pct_from_baseline = (max_kg / float(baseline_kg)) * 100.0
                wt_selected = fmt_pct(pct_from_baseline)
                if wt_selected not in wt_llm_extracted:
                    wt_llm_extracted.insert(0, wt_selected)
        elif chosen_from == "percent":
            if not wt_selected:
                if wt_llm_extracted:
                    wt_selected = wt_llm_extracted[0]
                else:
                    pct_vals = []
                    for m in re_pct.finditer(weight_sentence_str):
                        v = parse_number(m.group(1))
                        if not math.isnan(v):
                            pct_vals.append(v)
                    if pct_vals:
                        wt_selected = fmt_pct(max(pct_vals))

        # Normalize selected
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
            'weight_pref': wt_pref,
            'hba_debug': hba_debug,
            'wt_debug': wt_debug,
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

# -------------------- Quick LLM test UI --------------------
if configure_error:
    st.sidebar.error(f"Gemini configure: {configure_error}")

if st.sidebar.button("Run quick LLM test on sample sentence"):
    sample = "Sample sentence: HbA1c decreased by 1.2%."
    t, err = call_model_deterministic(_model_handle, "Return exactly a short JSON confirming you saw: " + sample)
    st.sidebar.text_area("Model raw output", value=t or "(empty)", height=160)
    if err:
        st.sidebar.warning("Model error: " + str(err))

# -------------------- Run processing --------------------
out_df = process_df(_model_handle if use_llm else None, df, col_name, drug_col, baseline_weight_kg)

# -------------------- Reorder columns --------------------
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
cols = insert_after(cols, "weight_extracted_matches", ["Weight LLM extracted", "weight KG", "Weight selected %", "Weight Score", "weight_pref"])
# include debug columns at end if show_debug
if show_debug:
    for dcol in ['hba_debug', 'wt_debug']:
        if dcol not in cols:
            cols.append(dcol)
display_df = display_df[[c for c in cols if c in display_df.columns]]

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
    file_name='results_strict_debug_fallback_llm_fixed.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
