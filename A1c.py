# streamlit_a1c_duration_extractor_with_hardcoded_key.py
"""
Streamlit app: HbA1c (A1c) extraction + scoring + duration column.
Includes optional LLM (Gemini) extraction for A1c and duration. API key is hard-coded below.
Usage: streamlit run streamlit_a1c_duration_extractor_with_hardcoded_key.py
"""

import re
import math
import json
from io import BytesIO
import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
API_KEY = "REPLACE_WITH_YOUR_REAL_GEMINI_API_KEY"
# =========================================================================

# Optional Gemini support (lazy import)
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c + Duration Extractor (with LLM)", layout="wide")
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
re_reduction_cue = re.compile(
    r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b',
    FLAGS
)

# Duration regex
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
    parts = re.split(r'(?<=[\.!\?])\s+|\n+', text)
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

# Window helper (±5 spaces)
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

# -------------------- Core regex extraction for A1c --------------------
def extract_in_sentence_regex(sent: str, si: int):
    matches = []
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

def sentence_meets_criterion_regex(sent: str) -> bool:
    has_term = bool(re_hba1c.search(sent))
    has_pct = bool(re_pct.search(sent))
    has_cue = bool(re_reduction_cue.search(sent))
    return has_term and has_pct and has_cue

# -------------------- LLM extraction for A1c + duration --------------------
LLM_RULES_A1C_DURATION = r"""
You are an information-extraction assistant. Read the supplied SENTENCE(S) and extract change magnitudes relevant to HbA1c/A1c and any trial/timepoint duration that applies to the reported result.
Return EXACTLY one JSON object and nothing else.

Allowed keys:
{
  "extracted": ["1.23%"],          // array of candidate percent magnitudes (strings, may be empty)
  "selected_percent": "1.23%",    // OPTIONAL: best single percent (positive, with %)
  "duration": "12 months",        // OPTIONAL: duration/timepoint associated with the result (e.g., "T12", "12 months", "6 weeks")
  "confidence": 0.87                // OPTIONAL: confidence 0.0-1.0
}

Rules:
- Percent strings must use dot decimal and end with '%', e.g., "1.75%".
- Duration strings should be human-readable (examples: "T12", "12 months", "6-12 weeks", "26 weeks").
- If you detect a `from X% to Y%` phrase, compute relative reduction percent using ((X - Y) / Y) * 100 and include it among "extracted" AND set "selected_percent" to the relative if it is the clearest representation.
- Do NOT emit any other text or commentary. Output must be valid JSON.
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

def llm_extract_a1c_and_duration(model, sentences: str):
    """Call LLM to extract JSON. Returns (extracted_list, selected_percent, duration_str) or ([], '', '') on failure."""
    if model is None or not sentences or not sentences.strip():
        return [], "", ""

    prompt = (
        "SENTENCES:\n" + sentences + "\n\n" + LLM_RULES_A1C_DURATION + "\nReturn JSON only."
    )
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
                    # normalize
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
@st.cache_data
def process_df(df_in: pd.DataFrame, text_col: str, model, use_llm: bool):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, '')
        if not isinstance(text_orig, str):
            text_orig = '' if pd.isna(text_orig) else str(text_orig)

        # Duration by regex (always available)
        duration_regex = extract_durations_regex(text_orig)

        # Regex-based A1c
        hba_matches, hba_sentences = [], []
        for si, sent in enumerate(split_sentences(text_orig)):
            if not sentence_meets_criterion_regex(sent):
                continue
            hba_sentences.append(sent)
            hba_matches.extend(extract_in_sentence_regex(sent, si))

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

        # LLM extraction (if enabled)
        llm_extracted = []
        llm_selected = ''
        llm_duration = ''
        sentences_joined = ' | '.join(hba_sentences) if hba_sentences else ''
        if use_llm and model and sentences_joined:
            llm_extracted, llm_selected, llm_duration = llm_extract_a1c_and_duration(model, sentences_joined)

        # Preference: if LLM gave a duration use it, else use regex duration
        final_duration = llm_duration if llm_duration else duration_regex

        # Choose selected percent: prefer LLM selected, else precomputed relative from regex, else first percent from regex
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
                    s2 = it.replace('%', '').replace(',', '.').strip()
                    try:
                        v = float(s2)
                        selected = fmt_pct(abs(v))
                        break
                    except:
                        continue

        a1c_score = compute_a1c_score(selected)

        new = row.to_dict()
        new.update({
            'sentence': sentences_joined,
            'extracted_matches': hba_regex_vals,
            'LLM extracted': llm_extracted,
            'selected %': selected,
            'A1c Score': a1c_score,
            'duration': final_duration,
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # keep rows that have extraction (regex or LLM)
    def has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask = (out['sentence'].astype(str).str.len() > 0) & ((out['extracted_matches'].apply(has_items)) | (out['LLM extracted'].apply(has_items)))
    out = out[mask].reset_index(drop=True)

    out.attrs['counts'] = dict(kept=int(mask.sum()), total=int(len(out)))
    return out

# -------------------- UI --------------------
st.sidebar.header('Options')
use_llm = st.sidebar.checkbox('Enable Gemini LLM (Gemini 2.0 Flash) for A1c + duration', value=False)
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
show_debug = st.sidebar.checkbox('Show debug columns (extracted_matches, LLM extracted, sentence)', value=True)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
    st.stop()

try:
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, sheet_name=0)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

# Configure model
model = None
if use_llm:
    if not GENAI_AVAILABLE:
        st.warning('google.generativeai (Gemini SDK) not available — LLM disabled.')
        use_llm = False
    else:
        model = configure_gemini(API_KEY)
        if model is None:
            st.warning('Failed to configure Gemini model — check API key and environment. LLM disabled.')
            use_llm = False

st.success(f'Loaded {len(df)} rows. Processing...')

out_df = process_df(df, col_name, model, use_llm)

# Reorder columns so duration is last
if 'duration' in out_df.columns:
    cols_no_duration = [c for c in out_df.columns if c != 'duration']
    cols_no_duration.append('duration')
    out_df = out_df[cols_no_duration]

st.write("### Results (first 200 rows shown)")
if not show_debug:
    for c in ['extracted_matches', 'LLM extracted', 'sentence']:
        if c in out_df.columns:
            out_df = out_df.drop(columns=[c])

st.dataframe(out_df.head(200))

counts = out_df.attrs.get('counts', None)
if counts:
    kept = counts.get('kept', 0)
    total = counts.get('total', 0)
    st.caption(f"Kept {kept} rows with A1c extraction (from {total} processed rows).")

# Download
@st.cache_data
def to_excel_bytes(df_out):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(out_df)
st.download_button('Download results as Excel', data=excel_bytes, file_name='a1c_results_with_duration_and_llm.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
