import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== HARD-CODED GEMINI 2.0 FLASH API KEY =====================
# (you gave this earlier; replace if needed)
API_KEY = "WEJKEBHABLRKJVBR;KEARVBBVEKJ"
# ==============================================================================

# Lazy import so the app still runs if package missing
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + sentence-level LLM (Gemini 2.0 Flash)")

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

# Terms & cues
re_hba1c  = re.compile(r'\bhb\s*a1c\b|\bhba1c\b|\ba1c\b', FLAGS)
re_weight = re.compile(r'\b(body\s*weight|weight|bw)\b', FLAGS)
re_reduction_cue = re.compile(
    r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b',
    FLAGS
)
# Timepoint cues
re_t12 = re.compile(r'\b(t12|12\s*months?|12-mo|month\s*12)\b', FLAGS)
re_t6  = re.compile(r'\b(t6|6\s*months?|6-mo|month\s*6)\b', FLAGS)

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

# --- build a local window by counting spaces (INCLUDE bordering tokens) ---
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

# -------------------- Core extraction --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []

    # 1) WHOLE-SENTENCE: from X% to Y%  -> X - Y (pp)
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red)

    # 2) ±5-SPACES window (inclusive) around term: other patterns only
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

    # 3) Weight safety fallback: nearest previous % within 60 chars if no window hit
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

# -------------------- Gemini helpers (reads ONLY 'sentence' cols) --------------------
def configure_gemini():
    if not GENAI_AVAILABLE:
        return None
    try:
        genai.configure(api_key=API_KEY)
        # Low temperature to reduce hallucinations
        return genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.2})
    except Exception:
        return None

LLM_RULES = (
    "You are a medical abstract assistant. Extract ONLY percentage changes relevant to the target metric.\n"
    "- Inputs: a sentence (or pipe-joined sentences) and a list of candidate percentage strings.\n"
    "- Candidates are the ONLY values you may output.\n"
    "- If timepoints are mentioned, prefer T12 / 12 months over T6 / 6 months. If neither, pick the main reported change.\n"
    "- Output strictly JSON: {\"extracted\": [<subset of candidates>], \"selected_percent\": <one of extracted>} with '%' signs.\n"
    "- Do NOT invent values. Do NOT include commentary."
)

def llm_select(model, label: str, sentence_text: str, candidates: list[str]) -> tuple[list[str], str]:
    """
    label: 'HbA1c' or 'Weight'
    sentence_text: string from the 'sentence' (or 'weight_sentence') column
    candidates: regex candidates from that sentence (strings like '1.2%')
    Returns (extracted_list, selected_percent) -- both restricted to the candidates set.
    """
    if model is None or not sentence_text or not candidates:
        return candidates, (candidates[0] if candidates else "")

    # Timepoint hint to make it explicit
    has_t12 = bool(re_t12.search(sentence_text))
    has_t6  = bool(re_t6.search(sentence_text))
    time_hint = "Prefer T12/12 months." if has_t12 else ("Prefer T6/6 months." if has_t6 else "No explicit timepoint preference mentioned.")

    prompt = f"""Task: Extract {label} percentage change from the sentence. Use only the provided candidates.

{LLM_RULES}

Sentence:
{sentence_text}

Timepoint hint: {time_hint}

Candidates:
{json.dumps(candidates)}

Return ONLY JSON like:
{{"extracted":["1.2%","0.8%"], "selected_percent":"1.2%"}}
"""

    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # find JSON
        s = text.find("{"); e = text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = data.get("extracted", [])
            selected = data.get("selected_percent", "")

            # Normalize & restrict to candidates
            def norm(v: str) -> str:
                v = (v or "").strip().replace(" ", "")
                if v and not v.endswith("%"):
                    if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
                        v += "%"
                return v

            cand_set = {norm(c) for c in candidates}
            extracted_norm = [norm(x) for x in extracted if norm(x) in cand_set]
            # if extracted empty, fall back to all candidates
            if not extracted_norm:
                extracted_norm = [norm(c) for c in candidates]

            selected_norm = norm(selected)
            if selected_norm not in cand_set:
                # apply timepoint preference heuristics if present, else first candidate
                if has_t12:
                    # choose the numerically closest to any value near phrases "12 months"/"T12" in sentence, else first of extracted
                    # (pragmatic fallback: first of extracted)
                    selected_norm = extracted_norm[0]
                elif has_t6:
                    selected_norm = extracted_norm[0]
                else:
                    selected_norm = extracted_norm[0]

            return extracted_norm, selected_norm
    except Exception:
        pass

    # fallback
    return candidates, candidates[0] if candidates else ""

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

if not uploaded:
    st.info('Upload your Excel/CSV (column name like "abstract").')
    st.stop()

# read file robustly
try:
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else None)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

st.success(f'Loaded {len(df)} rows. Processing regex…')

@st.cache_data
def process_regex(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        text = '' if not isinstance(text, str) else text

        # HbA1c extraction (regex -> strict filters)
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')

        # STRICT FILTER: keep only <7 pp (prefer reduction_pp if present)
        def allowed_hba(m):
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                return float(rp) < 7.0
            for v in (m.get('values') or []):
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) < 7.0:
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

        # Weight extraction (no <7 cutoff)
        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')

        def fmt_extracted_wt(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        new = row.to_dict()
        # HbA1c cols
        new.update({
            'sentence': ' | '.join(hba_sentences) if hba_sentences else '',
            'extracted_matches': [fmt_extracted_hba(m) for m in hba_matches],
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
        })
        # Weight cols
        new.update({
            'weight_sentence': ' | '.join(wt_sentences) if wt_sentences else '',
            'weight_extracted_matches': [fmt_extracted_wt(m) for m in wt_matches],
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # Keep row if HbA1c qualifies OR Weight qualifies
    def _list_has_items(x):
        return isinstance(x, list) and len(x) > 0
    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(_list_has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(_list_has_items))
    out = out[mask_hba | mask_wt].reset_index(drop=True)
    return out

regex_df = process_regex(df, col_name)

st.success("Regex done. Running Gemini on the sentence columns now (restricted to regex candidates)…")
model = configure_gemini()

# ---------------- LLM pass (ONLY sentence columns; ONLY from candidates) ----------------
def apply_llm_selection(df_in: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df_in.iterrows():
        sentence = row.get('sentence', '') or ''
        candidates_hba = row.get('extracted_matches', []) or []
        weight_sentence = row.get('weight_sentence', '') or ''
        candidates_wt = row.get('weight_extracted_matches', []) or []

        # HbA1c: restrict to candidates; prefer T12 > T6 if present in sentence
        hba_llm_list, hba_sel = llm_select(model, "HbA1c", sentence, candidates_hba)
        # Weight: same logic
        wt_llm_list, wt_sel = llm_select(model, "Weight", weight_sentence, candidates_wt)

        new = dict(row)
        new.update({
            'LLM extracted': hba_llm_list,
            'selected %': hba_sel,
            'Weight LLM extracted': wt_llm_list,
            'Weight selected %': wt_sel,
        })
        rows.append(new)
    return pd.DataFrame(rows)

out_df = apply_llm_selection(regex_df)

# ---------------- Display & download ----------------
st.write('### Results (first 200 rows)')
display_df = out_df.copy()
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])
st.dataframe(display_df.head(200))

@st.cache_data
def to_excel_bytes(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df.to_excel(w, index=False)
    buf.seek(0)
    return buf.getvalue()

st.download_button(
    'Download results as Excel',
    data=to_excel_bytes(out_df),
    file_name='results_with_hba1c_weight_llm_gemini2.0.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

st.caption("LLM picks are restricted to regex candidates in the sentence. Timepoint preference: T12/12 months > T6/6 months when present.")
