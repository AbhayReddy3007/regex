# streamlit_a1c_duration_extractor_with_llm.py
"""
Streamlit app: HbA1c (A1c) extraction + scoring + duration column.
Includes optional Gemini LLM extraction for A1c and duration. The LLM will read **only** sentences that meet the criterion:
  1) mentions A1c (A1c/HbA1c/Hb A1c),
  2) contains a reduction cue (reduce/decrease/decline/fell/etc.), and
  3) contains a numeric percent with a literal '%' sign.

This file is a drop-in enhanced version and keeps the processing function cache-safe (model parameter named `_model`).

Usage: streamlit run streamlit_a1c_duration_extractor_with_llm.py
"""

import re
import math
import json
from io import BytesIO
import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE (optional) =====================
# If you prefer to hardcode, put your key here. Otherwise leave empty and provide via sidebar.
API_KEY = ""
# ================================================================================

# Lazy Gemini import so the app still runs without the package
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c + Duration Extractor (LLM)", layout="wide")
st.title("HbA1c (A1c) — regex + optional Gemini LLM extraction + score + duration")

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
re_reduction_cue = re.compile(r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b', FLAGS)

# Duration regex (fallback)
DURATION_RE = re.compile(
    r'\b(?:T\d{1,2}|'                                       # T6, T12
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:weeks?|wks?|wk|w)\b|'   # 12 weeks, 6-12 weeks
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:months?|mos?|mo)\b|'    # 6 months, 12-mo, 6-12 months
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:days?|d)\b|'            # days
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:years?|yrs?)\b|'        # years
    r'\d{1,3}-week\b|\d{1,3}-month\b|\d{1,3}-mo\b)',         # hyphenated forms
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
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]


def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.2f}".rstrip('0').rstrip('.')
    return f"{s}%"


def extract_durations_regex(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    found = []
    seen = set()
    for m in DURATION_RE.finditer(text):
        token = m.group(0).strip()
        token = re.sub(r'\s+', ' ', token)
        token = token.replace('–', '-').replace('—', '-')
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

# small window builder used to search near the target term (±5 spaces inclusive tokens)
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
        'raw': m.group(0) if hasattr(m, 'group') else str(m),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (abs_start + (m.start() if hasattr(m, 'start') else 0), abs_start + (m.end() if hasattr(m, 'end') else 0)),
    })

# -------------------- Core extraction (regex) --------------------
def extract_in_sentence_regex(sent: str, si: int):
    matches = []

    # 1) WHOLE-SENTENCE: from X% to Y%
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red_pp = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        rel = None
        if not (math.isnan(a) or math.isnan(b)) and b != 0:
            rel = ((a - b) / b) * 100.0
        add_match(matches, si, 0, m, 'from-to_sentence', [a, b], red_pp)
        if rel is not None:
            rel_raw = f"{rel:.6f}%"
            matches.append({
                'raw': rel_raw,
                'type': 'from-to_relative_percent',
                'values': [a, b, rel],
                'reduction_pp': red_pp,
                'sentence_index': si,
                'span': (m.start(), m.end()),
            })

    # 2) ±5-SPACES window around each A1c occurrence: other patterns only
    any_window_hit = False
    for hh in re_hba1c.finditer(sent):
        seg, abs_s, _ = window_prev_next_spaces_inclusive_tokens(sent, hh.end(), 5, 5)

        for m in re_reduce_by.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, 'percent_or_pp_pmSpaces5', [v], v)

        for m in re_abs_pp.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, 'pp_word_pmSpaces5', [v], v)

        for m in re_range.finditer(seg):
            any_window_hit = True
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            rep = None if (math.isnan(a) or math.isnan(b)) else max(a, b)
            add_match(matches, si, abs_s, m, 'range_percent_pmSpaces5', [a, b], rep)

        for m in re_pct.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, 'percent_pmSpaces5', [v], v)

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


def sentence_meets_criterion(sent: str) -> bool:
    """Return True only if the sentence contains all three:
      1) an A1c mention (A1c/HbA1c/Hb a1c variants),
      2) a reduction cue word (reduce/decrease/etc.), and
      3) a numeric percent with a literal '%' sign (not a cutoff like '>=5%').
    """
    has_term = bool(re_hba1c.search(sent))
    # require a literal percent sign and avoid relational operators immediately adjacent
    has_pct = bool(re.search(r"(?<![<>≥≤])" + PCT, sent))
    # reduction cue (reduce, decrease, decline, fell, etc.)
    has_cue = bool(re_reduction_cue.search(sent))
    return bool(has_term and has_pct and has_cue)


def extract_sentences(text: str):
    matches, sentences_used = [], []
    for si, sent in enumerate(split_sentences(text)):
        if not sentence_meets_criterion(sent):
            continue
        sentences_used.append(sent)
        matches.extend(extract_in_sentence_regex(sent, si))

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

# -------------------- LLM rules & helpers --------------------
LLM_RULES = """
You are an extraction assistant. Input: a set of SENTENCES that each mention HbA1c/A1c, a reduction cue, and a percent.
Return exactly one JSON object, no extra text.
Allowed keys:
{
  "extracted": ["1.75%", "2.0%"],
  "selected_percent": "1.75%",   // optional
  "duration": "12 months",       // optional
  "confidence": 0.9                // optional
}
Rules:
- Percent strings must use '.' decimal and end with '%'.
- Duration should be human readable (e.g., 'T12', '12 months', '26 weeks').
- If a 'from X% to Y%' is present, compute relative reduction ((X - Y) / Y) * 100 and include it.
- Only extract change magnitudes (do not extract thresholds like '>=5%').
- Return strict JSON only.
"""


def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None


def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v


def llm_extract_from_sentences(model, sentences: str):
    """Call LLM with the short set of sentences. Returns (extracted_list, selected_percent, duration)
    On failure returns ([], '', '')."""
    if model is None or not sentences or not sentences.strip():
        return [], "", ""
    prompt = "SENTENCES:\n" + sentences + "\n\n" + LLM_RULES + "\nReturn JSON only."
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find('{'), text.rfind('}')
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = []
            for x in (data.get('extracted') or []):
                if not isinstance(x, str):
                    continue
                x2 = _norm_percent(x)
                if re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', x2):
                    n = parse_number(x2.replace('%', ''))
                    if not math.isnan(n):
                        extracted.append(fmt_pct(abs(n)))
            selected = _norm_percent(data.get('selected_percent', '') or "")
            if selected:
                try:
                    n = parse_number(selected.replace('%', ''))
                    if not math.isnan(n):
                        selected = fmt_pct(abs(n))
                except Exception:
                    selected = ''
            duration = (data.get('duration') or '').strip()
            return extracted, selected, duration
    except Exception:
        return [], "", ""
    return [], "", ""

# -------------------- Scoring --------------------

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

# -------------------- Processing --------------------
# Note: rename model argument to _model so Streamlit's cache does not try to hash it.
@st.cache_data
def process_df(df_in: pd.DataFrame, text_col: str, _model, use_llm: bool):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, '')
        if not isinstance(text_orig, str):
            text_orig = '' if pd.isna(text_orig) else str(text_orig)

        # Extract regex durations as fallback
        duration_regex = extract_durations_regex(text_orig)

        # Extract qualifying sentences and regex matches
        hba_matches, hba_sentences = extract_sentences(text_orig)
        sentences_joined = ' | '.join(hba_sentences) if hba_sentences else ''

        # Format regex outputs
        def fmt_extracted(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                if 'relative' in t:
                    vals = m.get('values') or []
                    if len(vals) >= 3 and vals[2] is not None and not math.isnan(vals[2]):
                        return fmt_pct(vals[2])
                    if m.get('reduction_pp') is not None and not math.isnan(m.get('reduction_pp')):
                        return fmt_pct(m.get('reduction_pp'))
                    return m.get('raw', '')
                else:
                    if m.get('reduction_pp') is not None and not math.isnan(m.get('reduction_pp')):
                        return fmt_pct(m.get('reduction_pp'))
                    return m.get('raw', '')
            if isinstance(m.get('reduction_pp'), (int, float)) and not math.isnan(m.get('reduction_pp')):
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        hba_regex_vals = [fmt_extracted(m) for m in hba_matches]

        # LLM extraction (only read the qualifying sentences)
        llm_extracted, llm_selected, llm_duration = [], '', ''
        if use_llm and _model and sentences_joined:
            llm_extracted, llm_selected, llm_duration = llm_extract_from_sentences(_model, sentences_joined)

        # Duration: prefer LLM duration if present
        final_duration = llm_duration if llm_duration else duration_regex

        # Determine selected percent: priority = LLM.selected -> precomputed relative (regex) -> first regex percent
        def _find_precomputed_relative(matches):
            for m in matches:
                t = (m.get('type') or '').lower()
                if 'from-to_relative_percent' in t or 'from-to_relative' in t:
                    vals = m.get('values') or []
                    if len(vals) >= 3 and vals[2] is not None and not math.isnan(vals[2]):
                        return fmt_pct(vals[2])
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

        precomputed_rel = _find_precomputed_relative(hba_matches)

        selected = ''
        if llm_selected:
            selected = llm_selected
        elif precomputed_rel:
            selected = precomputed_rel
        else:
            for it in hba_regex_vals:
                if isinstance(it, str) and it.strip().endswith('%'):
                    s2 = it.replace('%', '').replace(',', '.')
