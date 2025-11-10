import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# ================== HARD-CODED GEMINI API KEY ==================
API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"   # <-- replace with your real key
# ===============================================================

# -------------------- Optional Gemini imports (lazy) --------------------
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — whole-sentence from→to, ±5-spaces window for others + Gemini assist")

# -------------------- Regex helpers --------------------
# Allow optional sign and decimal separators '.', ',' or '·'
NUM = r'(?:[+-]?\d+(?:[.,·]\d+)?)'
PCT = rf'({NUM})\s*%'
DASH = r'(?:-|–|—)'  # '-', en dash, em dash

# Reduction patterns (all require %)
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
    """
    Return (segment, abs_start, abs_end) around index 'pos' spanning up to:
      - PREVIOUS n_prev_spaces whitespace boundaries (treat runs as one),
        and INCLUDE the full token immediately BEFORE that leftmost boundary,
      - NEXT n_next_spaces whitespace boundaries to the right,
        and INCLUDE the full token immediately AFTER that rightmost boundary.
    """
    space_like = set([' ', '\t', '\n', '\r'])
    L = len(s)

    # Left
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

    # Right
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

    # 1) whole sentence: from X% to Y% -> delta
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red)

    # 2) ±5-space window for other patterns
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

    # 3) weight safety: nearest previous % within 60 chars if window empty
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

# -------------------- Gemini helpers --------------------
def configure_gemini():
    if not GENAI_AVAILABLE or not API_KEY:
        return None
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        return None

LLM_SYSTEM_RULES = (
    "You extract ONLY HbA1c/A1c or body weight percentage changes from the provided sentences.\n"
    "- Prefer 'from X% to Y%' deltas as X - Y (percentage points).\n"
    "- You MAY propose a cleaned/normalized value even if it does NOT appear in the regex candidates (use the sentences).\n"
    "- If multiple values are present, choose the single best value that represents the reported change.\n"
    "- Output strictly in JSON with keys: extracted (array of strings like '1.2%'), selected_percent (string like '1.2%').\n"
    "- Do not include any commentary."
)

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v

def llm_choose_best(model, target_label: str, sentences: list, regex_candidates: list):
    """
    target_label: 'HbA1c' or 'Weight'
    sentences: list of strings (qualifying sentences)
    regex_candidates: list of strings extracted via regex.
    Returns (extracted_list, selected_percent_str) or ([], '')
    """
    if model is None:
        return [], ""

    joined_sent = "\n".join(f"- {s}" for s in sentences if s)
    joined_cand = "\n".join(f"- {c}" for c in regex_candidates if c)
    prompt = f"""Role: {target_label} percentage change extractor.

{LLM_SYSTEM_RULES}

Sentences:
{joined_sent if joined_sent else '- (none)'}

Regex candidates (HINTS only — you may output other cleaned values):
{joined_cand if joined_cand else '- (none)'}

Return JSON only. Example:
{{"extracted": ["1.3%","0.8%"], "selected_percent": "1.3%"}}
"""

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            js = text[start:end+1]
            data = json.loads(js)
            extracted = [_norm_percent(x) for x in data.get("extracted", []) if isinstance(x, str)]
            sel = _norm_percent(data.get("selected_percent", ""))
            return extracted, sel
    except Exception:
        pass

    # Fallbacks if LLM fails: use regex candidates (if any)
    rc_norm = [_norm_percent(x) for x in regex_candidates]
    sel = rc_norm[0] if rc_norm else ""
    return rc_norm, sel

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

st.sidebar.markdown("---")
use_llm = st.sidebar.checkbox("Enable Gemini 2.5 Flash for 'LLM extracted' & 'selected %'", value=True)

model = configure_gemini() if use_llm else None
if use_llm and model is None:
    st.sidebar.error("Gemini not available or API key invalid. LLM columns will remain empty.")

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
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

st.success(f'Loaded {len(df)} rows. Processing...')

@st.cache_data
def process_df(df, text_col, model):
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        text = '' if not isinstance(text, str) else text

        # ---- HbA1c extraction (strict rules & filtering) ----
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')

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

        # ---- Weight extraction ----
        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')

        def fmt_extracted_wt(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        # Prepare row dict
        new = row.to_dict()

        # HbA1c columns (regex)
        hba_regex_vals = [fmt_extracted_hba(m) for m in hba_matches]
        new.update({
            'sentence': ' | '.join(hba_sentences) if hba_sentences else '',
            'extracted_matches': hba_regex_vals,
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
        })

        # Weight columns (regex)
        wt_regex_vals = [fmt_extracted_wt(m) for m in wt_matches]
        new.update({
            'weight_sentence': ' | '.join(wt_sentences) if wt_sentences else '',
            'weight_extracted_matches': wt_regex_vals,
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
        })

        # --------------- LLM columns (optional) ---------------
        # HbA1c LLM
        hba_llm_extracted, hba_selected = ([], "")
        if model is not None and (hba_sentences or hba_regex_vals):
            hba_llm_extracted, hba_selected = llm_choose_best(model, "HbA1c", hba_sentences, hba_regex_vals)

        # Weight LLM
        wt_llm_extracted, wt_selected = ([], "")
        if model is not None and (wt_sentences or wt_regex_vals):
            wt_llm_extracted, wt_selected = llm_choose_best(model, "Weight", wt_sentences, wt_regex_vals)

        new.update({
            'LLM extracted': hba_llm_extracted,
            'selected %': hba_selected,
            'Weight LLM extracted': wt_llm_extracted,
            'Weight selected %': wt_selected,
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

    # Counts for UI
    out.attrs['counts'] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum()),
    )

    return out

out_df = process_df(df, col_name, model)

# display
st.write('### Results (first 200 rows shown)')
display_df = out_df.copy()
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

# download
@st.cache_data
def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(out_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_hba1c_weight_llm.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

st.markdown('---')
st.write("**Rules enforced:**")
st.write("- 'from X% to Y%' searched across the whole sentence; shown as delta (X−Y).")
st.write("- Other % patterns must be within the **previous 5 spaces** and **next 5 spaces** of the target term (including bordering tokens).")
st.write("- HbA1c values strictly **< 7**; weight has no numeric cutoff.")
st.write("- Row kept if **either** HbA1c **or** weight qualifies.")
st.write("- Gemini picks `LLM extracted` and a single `selected %` per target; it **may propose a cleaned value not in regex candidates**.")
