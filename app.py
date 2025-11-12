# streamlit_hba1c_weight_llm.py
import re
import math
import json
from io import BytesIO
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

# ===================== GEMINI KEY (insert locally, DO NOT COMMIT) =====================
API_KEY = ""   # <-- PUT YOUR GEMINI KEY HERE (e.g. "ya29...."). I cannot hardcode a user's secret.
# ================================================================================

# Lazy import for Gemini client
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
    if not isinstance(text, str):
        return []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(?<=[\.!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    return bool(term_re.search(sent)) and bool(re_pct.search(sent)) and bool(re_reduction_cue.search(sent))

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

def _norm_percent_str(v: str) -> str:
    """Normalize string like '1.2 %' -> '1.2%'. If plain number, append %."""
    if not v:
        return ""
    s = str(v).strip().replace(" ", "").replace("·", ".")
    if s.endswith('%'):
        return s
    if re.match(r'^[+-]?\d+(?:[.,]\d+)?$', s):
        return s + '%'
    return s

# small helper to reject likely non-effect percentages (p-values, SE, CI)
def looks_like_pvalue_or_ci(token_span: Tuple[int,int], sentence: str) -> bool:
    left = max(0, token_span[0]-6)
    context = sentence[left:token_span[1]+6].lower()
    # common markers to reject
    if re.search(r'\bse\b|\bci\b|\bbetween-group\b|\bp\s*[<=>]\b', context):
        return True
    # if preceded by '(' and contains SE inside parentheses, reject
    paren = sentence[max(0, token_span[0]-10): token_span[1]+10]
    if re.search(r'\bse\b|\bci\b', paren, flags=re.IGNORECASE):
        return True
    return False

# -------------------- Window selection (spaces) --------------------
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

# -------------------- Core extraction (unchanged logic) --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []

    # whole sentence from->to
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red)

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

# -------------------- UI: upload + options --------------------
st.sidebar.header('Options')
uploaded = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name_input = st.sidebar.text_input('Column with abstracts/text (exact name)', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (leave blank to use first sheet)', value='')
use_llm = st.sidebar.checkbox('Enable Gemini LLM extraction (gemini-2.0-flash)', value=True)
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)

if not uploaded:
    st.info('Upload a file (Excel or CSV) and enter the column name that contains abstracts (e.g. "abstract").')
    st.stop()

# -------------------- Read file robustly (avoid dict return) --------------------
try:
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        # for Excel, use the provided sheet name OR first sheet (sheet_name default None -> first sheet)
        if sheet_name:
            df = pd.read_excel(uploaded, sheet_name=sheet_name)
        else:
            df = pd.read_excel(uploaded)  # returns a DataFrame (first sheet)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

# validate chosen column
if not col_name_input or col_name_input not in df.columns:
    st.error(f'Column "{col_name_input}" not found. Available columns: {list(df.columns)}')
    st.stop()

st.success(f'Loaded {len(df)} rows. Processing...')

# -------------------- Regex-only processing (keeps sentence columns) --------------------
@st.cache_data
def process_df_regex_only(df: pd.DataFrame, text_col: str):
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        text = '' if not isinstance(text, str) else text

        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')

        def allowed_hba(m):
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                return float(abs(rp)) < 7.0  # allow negative deltas but filter by magnitude
            for v in (m.get('values') or []):
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and abs(float(v)) < 7.0:
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

        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')

        def fmt_extracted_wt(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        new = row.to_dict()
        # HbA1c columns
        new.update({
            'sentence': ' | '.join(hba_sentences) if hba_sentences else '',
            'extracted_matches': [fmt_extracted_hba(m) for m in hba_matches],
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
        })
        # Weight columns
        new.update({
            'weight_sentence': ' | '.join(wt_sentences) if wt_sentences else '',
            'weight_extracted_matches': [fmt_extracted_wt(m) for m in wt_matches],
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
        })
        rows.append(new)

    out = pd.DataFrame(rows)

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

out_df = process_df_regex_only(df, col_name_input)

# -------------------- Gemini helpers --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

model = configure_gemini(API_KEY) if use_llm else None

# LLM instruction (strict)
LLM_RULES = (
    "You are an extractor. INPUT = one sentence (or several joined by ' | ').\n"
    "TARGET specifies which target to extract: 'HbA1c' or 'Body weight'.\n"
    "Return only percentages that are the EFFECT on the TARGET (do NOT return p-values, SE, CI, between-group differences, or unrelated percentages).\n"
    "If 'from X% to Y%' present, compute change = X - Y (percentage points) and include it as a percentage (e.g. '1.3%').\n"
    "If multiple timepoints appear, prefer 12-month/T12 over 6-month/T6. Keywords: '12 months','12-mo','T12' > '6 months','6-mo','T6'.\n"
    "Output STRICT JSON only: {\"extracted\": [\"1.3%\",\"0.8%\"], \"selected_percent\": \"1.3%\"}\n"
    "Do not output any other text."
)

def llm_extract_from_sentence(model, target_label: str, sentence: str) -> Tuple[List[str], str]:
    """
    Returns (llm_extracted_list, selected_percent_str)
    """
    if model is None or not sentence.strip():
        return [], ""

    prompt = (
        f"TARGET: {target_label}\n\n"
        f"{LLM_RULES}\n\n"
        f"SENTENCE(S):\n{sentence}\n"
    )
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = [ _norm_percent_str(x) for x in (data.get("extracted") or []) if isinstance(x, str) ]
            selected = _norm_percent_str(data.get("selected_percent", "") or "")
            # Basic post-filter: remove percentages that look like p-values/SE/CI
            filtered = []
            for token in extracted:
                # find token span in sentence to check context; if multiple occurrences, accept the first non-rejected
                try:
                    idx = sentence.index(token)
                    if looks_like_pvalue_or_ci((idx, idx+len(token)), sentence):
                        continue
                except ValueError:
                    # token not found; still include (LLM might produce cleaned value)
                    pass
                filtered.append(token)
            if not filtered:
                # if filtered out all, fallback to raw extracted
                filtered = extracted
            # selected: ensure it's present in filtered; if not, try to pick best candidate
            if selected and selected not in filtered:
                # prefer a 12mo candidate if any
                pick = None
                for tok in filtered:
                    if re.search(r'(12\s*mo|12\s*months|t12)', sentence, flags=re.I) and re.search(r'(12\s*mo|12\s*months|t12)', sentence, flags=re.I):
                        pick = tok
                        break
                pick = pick or (filtered[0] if filtered else "")
                selected = pick
            # ensure positive in selected
            if selected:
                try:
                    val = float(selected.replace('%',''))
                    selected = fmt_pct(abs(val))
                except Exception:
                    selected = selected
            return filtered, selected
    except Exception:
        pass

    # fallback: return empty list & empty string
    return [], ""

# -------------------- Run LLM on sentence columns --------------------
# Use sentence columns produced by regex stage: 'sentence' and 'weight_sentence'
def run_llm_on_df(df_in: pd.DataFrame, model):
    df = df_in.copy()
    # HbA1c LLM
    hba_sent_series = df.get("sentence", pd.Series(dtype=str)).fillna("").astype(str)
    hba_pairs = hba_sent_series.apply(lambda s: llm_extract_from_sentence(model, "HbA1c", s))
    df["LLM extracted"] = [vals for vals, sel in hba_pairs]
    df["selected %"]   = [sel  for vals, sel in hba_pairs]

    # Weight LLM
    wt_sent_series = df.get("weight_sentence", pd.Series(dtype=str)).fillna("").astype(str)
    wt_pairs = wt_sent_series.apply(lambda s: llm_extract_from_sentence(model, "Body weight", s))
    df["Weight LLM extracted"] = [vals for vals, sel in wt_pairs]
    df["Weight selected %"]    = [sel  for vals, sel in wt_pairs]

    return df

out_df = run_llm_on_df(out_df, model)

# -------------------- Ensure selected % positive and formatted --------------------
def normalize_selected_pct(p):
    if not p:
        return ""
    try:
        num = float(str(p).replace('%','').replace(',','.'))
        return fmt_pct(abs(num))
    except:
        return str(p)

out_df["selected %"] = out_df["selected %"].apply(normalize_selected_pct)
out_df["Weight selected %"] = out_df["Weight selected %"].apply(normalize_selected_pct)

# -------------------- Scoring functions --------------------
def a1c_score_from_pct_str(pct_str: str) -> Optional[int]:
    if not pct_str:
        return None
    try:
        v = float(pct_str.replace('%','').replace(',','.'))
    except:
        return None
    # Scores for A1c
    if v > 2.2:
        return 5
    if 1.8 <= v <= 2.1:
        return 4
    if 1.2 <= v <= 1.7:
        return 3
    if 0.8 <= v <= 1.1:
        return 2
    if v < 0.8:
        return 1
    return None

def weight_score_from_pct_str(pct_str: str) -> Optional[int]:
    if not pct_str:
        return None
    try:
        v = float(pct_str.replace('%','').replace(',','.'))
    except:
        return None
    # Scores for weight
    if v >= 22:
        return 5
    if 16 <= v <= 21.9:
        return 4
    if 10 <= v <= 15.9:
        return 3
    if 5 <= v <= 9.9:
        return 2
    if v < 5:
        return 1
    return None

out_df["A1c Score"] = out_df["selected %"].apply(lambda x: a1c_score_from_pct_str(x))
out_df["Weight Score"] = out_df["Weight selected %"].apply(lambda x: weight_score_from_pct_str(x))

# -------------------- Reorder columns: place LLM columns beside regex columns --------------------
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
# keep only those columns that exist (safe)
cols = [c for c in cols if c in display_df.columns]
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
def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(display_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_llm_from_sentence.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
