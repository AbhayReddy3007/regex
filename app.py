# app.py
import os
import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# -------------------- Model config (Gemini 2.0 Flash) --------------------
GEMINI_MODEL_NAME = "gemini-2.0-flash"

GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

def configure_gemini_from_env():
    """
    Configure Gemini with GEMINI_API_KEY environment variable.
    Returns a GenerativeModel or None if not configured.
    """
    if not GENAI_AVAILABLE:
        return None
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception:
        return None

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Body Weight — whole-sentence from→to, ±5-spaces window + Gemini 2.0 Flash on sentence columns")

# -------------------- Regex helpers (unchanged core) --------------------
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

# --- window by counting spaces (inclusive of bordering tokens) ---
def window_prev_next_spaces_inclusive_tokens(s: str, pos: int, n_prev_spaces: int = 5, n_next_spaces: int = 5):
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

    # 1) whole-sentence from X% to Y% -> delta
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red)

    # 2) window ±5 spaces for other patterns
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

def extract_sentences(text: str, term_re: re.Pattern, tag_prefix: str):
    matches, sentences_used = [], []
    for si, sent in enumerate(split_sentences(text)):
        if not sentence_meets_criterion(sent, term_re):
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

# -------------------- UI inputs --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar.')
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

# -------------------- Phase 1: regex pipeline (creates sentence columns) --------------------
@st.cache_data
def process_regex(df, text_col):
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        text = '' if not isinstance(text, str) else text

        # HbA1c extraction
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')

        # strict filter HbA1c: keep only < 7 (prefer reduction_pp)
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

        # Weight extraction
        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')

        def fmt_extracted_wt(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        new = row.to_dict()
        new.update({
            # HbA1c
            'sentence': ' | '.join(hba_sentences) if hba_sentences else '',
            'extracted_matches': [fmt_extracted_hba(m) for m in hba_matches],
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
            # Weight
            'weight_sentence': ' | '.join(wt_sentences) if wt_sentences else '',
            'weight_extracted_matches': [fmt_extracted_wt(m) for m in wt_matches],
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # keep if either HbA1c or Weight qualifies (has non-empty extracted list and sentences)
    def _list_has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(_list_has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(_list_has_items))
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

regex_df = process_regex(df, col_name)

# -------------------- Phase 2: LLM over the sentence columns --------------------
def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%") and re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
        v += "%"
    return v

LLM_SYSTEM_RULES = (
    "You extract ONLY HbA1c/A1c or body weight percentage changes from the provided sentences.\n"
    "- Prefer 'from X% to Y%' deltas as X - Y (percentage points).\n"
    "- You MAY propose a cleaned/normalized value even if it does NOT appear in the hints.\n"
    "- If multiple values are present, choose the single best value that represents the reported change.\n"
    "- Output strictly JSON with keys: extracted (array of strings like '1.2%'), selected_percent (string like '1.2%').\n"
    "- Do not include any commentary."
)

def llm_extract_from_sentences(model, target_label: str, sentence_text: str, hints: list[str]):
    """
    sentence_text: the contents of the 'sentence' (or 'weight_sentence') column (pipe-separated).
    hints: regex hits (we pass as hints only; LLM can propose cleaned value).
    Returns (list_extracted, selected_percent_str)
    """
    if model is None or (not sentence_text or not sentence_text.strip()):
        return [], ""

    sent_list = [s.strip() for s in sentence_text.split("|") if s and s.strip()]
    joined_sent = "\n".join(f"- {s}" for s in sent_list)
    joined_hints = "\n".join(f"- {h}" for h in (hints or []))

    prompt = f"""Role: {target_label} percentage change extractor.

{LLM_SYSTEM_RULES}

Sentences:
{joined_sent if joined_sent else '- (none)'}

Hints (regex candidates; you may output other cleaned values if better):
{joined_hints if joined_hints else '- (none)'}

Return JSON only. Example:
{{"extracted": ["1.3%","0.8%"], "selected_percent": "1.3%"}}
"""
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        start = text.find("{"); end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
            ext = [_norm_percent(x) for x in data.get("extracted", []) if isinstance(x, str)]
            sel = _norm_percent(data.get("selected_percent", ""))
            return ext, sel
    except Exception:
        pass
    # fallback: just echo hints, choose first
    hints_norm = [_norm_percent(x) for x in (hints or [])]
    return hints_norm, (hints_norm[0] if hints_norm else "")

@st.cache_data
def run_llm_over_sentences(_df: pd.DataFrame):
    model = configure_gemini_from_env()
    # Prepare new columns even if model is missing
    out = _df.copy()

    # HbA1c LLM columns
    h_llm = []
    h_sel = []
    # Weight LLM columns
    w_llm = []
    w_sel = []

    for _, r in out.iterrows():
        ext_h, sel_h = llm_extract_from_sentences(
            model, "HbA1c",
            r.get("sentence", ""),
            r.get("extracted_matches", []),
        )
        h_llm.append(ext_h); h_sel.append(sel_h)

        ext_w, sel_w = llm_extract_from_sentences(
            model, "Weight",
            r.get("weight_sentence", ""),
            r.get("weight_extracted_matches", []),
        )
        w_llm.append(ext_w); w_sel.append(sel_w)

    out["LLM extracted"] = h_llm
    out["selected %"] = h_sel
    out["Weight LLM extracted"] = w_llm
    out["Weight selected %"] = w_sel

    # reorder so LLM cols sit next to regex cols
    def insert_after(cols, after, names):
        cols = list(cols)
        if after in cols:
            idx = cols.index(after)
            for name in names[::-1]:
                if name in cols:
                    cols.remove(name)
                cols.insert(idx + 1, name)
        return cols

    final_cols = list(out.columns)
    final_cols = insert_after(final_cols, "extracted_matches", ["LLM extracted", "selected %"])
    final_cols = insert_after(final_cols, "weight_extracted_matches", ["Weight LLM extracted", "Weight selected %"])
    out = out[final_cols]
    return out

out_df = run_llm_over_sentences(regex_df)

# -------------------- Display --------------------
st.write('### Results (first 200 rows shown)')
display_df = out_df.copy()
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])
st.dataframe(display_df.head(200))

# counts
counts = regex_df.attrs.get('counts', None)
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
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
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

# -------------------- Helpful diagnostics --------------------
if not GENAI_AVAILABLE:
    st.warning("google-generativeai not installed. Install with: pip install google-generativeai")
elif not os.environ.get("GEMINI_API_KEY"):
    st.warning("GEMINI_API_KEY is not set. LLM columns will use regex hints only. See instructions at the top.")
