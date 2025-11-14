# streamlit_hba1c_weight_llm_fixed.py
import re
import math
import json
import unicodedata
from io import BytesIO
from typing import List, Tuple

import pandas as pd
import streamlit as st

# -------------------- HARD-CODE YOUR GEMINI KEY HERE --------------------
API_KEY = "PASTE_YOUR_REAL_GEMINI_KEY_HERE"  # <<--- put your key here
# -----------------------------------------------------------------------

# Lazy import for Gemini so app still runs without the package
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor (fixed)", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (fixed)")

# -------------------- Regex helpers (unchanged logic) --------------------
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
def normalize_text(txt: str) -> str:
    """Normalize encoding artifacts and smart punctuation, return cleaned unicode string."""
    if not isinstance(txt, str):
        return ""
    # Replace known mojibake sequences, replace middle dot variants with dot
    txt = txt.replace('Â·', '.').replace('·', '.').replace('\xa0', ' ')
    # Remove soft hyphens and replace weird placeholders
    txt = txt.replace('\ufeff', '').replace('\ufffd', '')
    # Normalize unicode (NFKC) to unify strange punctuation
    txt = unicodedata.normalize("NFKC", txt)
    return txt

def parse_number(s: str) -> float:
    if s is None:
        return float('nan')
    s = str(s).strip().replace(',', '.').replace('·', '.')
    try:
        return float(s)
    except Exception:
        return float('nan')

def split_sentences(text: str) -> List[str]:
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

# window by spaces (same as your existing)
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

# -------------------- Core extraction (keeps your logic) --------------------
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
    # weight fallback nearest prev %
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
    # de-dupe
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
        if not (term_re.search(sent) and re_pct.search(sent) and re_reduction_cue.search(sent)):
            continue
        sentences_used.append(sent)
        matches.extend(extract_in_sentence(sent, si, term_re, tag_prefix))
    # dedupe
    seen, filtered = set(), []
    for mm in matches:
        key = (mm['sentence_index'], mm['span'])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(mm)
    filtered.sort(key=lambda x: (x['sentence_index'], x['span'][0]))
    return filtered, sentences_used

# -------------------- LLM helpers --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        # use the 2.0 flash model requested
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v

# LLM system prompt rules
LLM_RULES = (
    "You are an extractor. ONLY extract percentage change values that refer to the TARGET.\n"
    "- TARGET will be 'HbA1c' or 'Body weight'.\n"
    "- Read the provided SENTENCE(S) carefully and return JSON only.\n"
    "- IGNORE threshold/eligibility values like 'HbA1c < 7.0%' or '>=5%' (these are endpoints/thresholds, not reductions).\n"
    "- If sentence has within-group change and also between-group difference, prefer the within-group value for the named drug unless asked otherwise.\n"
    "- If multiple timepoints exist, prefer 12 months / T12 over 6 months / T6.\n"
    "- If multiple drugs appear, use the provided 'drug_name' context (if available) and pick the value related to that drug.\n"
    "- For 'from X% to Y%', compute X - Y and include that in extracted values (as a percentage string with %). Also you may include explicit 'reduced by Z%'.\n"
    "- Return JSON like: {\"extracted\": [\"1.75%\",\"0.4%\"], \"selected_percent\": \"1.75%\"}\n"
)

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name: str = "") -> Tuple[List[str], str]:
    """Return (extracted_list, selected_percent_str). Empty if no model or nothing extracted."""
    if model is None or not sentence or not sentence.strip():
        return [], ""
    prompt = (
        f"TARGET: {target_label}\n"
        f"DRUG_NAME: {drug_name}\n\n"
        f"{LLM_RULES}\n\n"
        f"SENTENCE(S):\n{sentence}\n"
    )
    try:
        # generate_content returns an object with .text in older client patterns
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # locate JSON object in response
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = [_norm_percent(x) for x in (data.get("extracted") or []) if isinstance(x, str)]
            selected = _norm_percent(data.get("selected_percent") or "")
            # ensure selected becomes empty if LLM extracted empty
            if not extracted:
                return [], ""
            # convert negative sign to positive for selected
            if selected.startswith("-"):
                selected = selected.replace("-", "")
                if selected and selected[0] != "-":
                    # ensure canonical formatting
                    selected = _norm_percent(selected)
            return extracted, selected
    except Exception:
        # fail silently back to empty
        pass
    return [], ""

# -------------------- Scoring functions --------------------
def a1c_score_from_percent(selected_pct: str) -> int:
    """Return score 1-5 for HbA1c using rules provided. selected_pct like '1.75%' or ''."""
    if not selected_pct:
        return None
    try:
        v = float(selected_pct.replace('%', '').replace(',', '.'))
    except:
        return None
    # Use absolute percent reduction for scoring
    if v > 2.2:
        return 5
    if 1.8 <= v <= 2.2:
        return 4
    if 1.2 <= v <= 1.7:
        return 3
    if 0.8 <= v <= 1.1:
        return 2
    if v < 0.8:
        return 1
    return None

def weight_score_from_percent(selected_pct: str) -> int:
    if not selected_pct:
        return None
    try:
        v = float(selected_pct.replace('%', '').replace(',', '.'))
    except:
        return None
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

# -------------------- UI / File handling --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
drug_col   = st.sidebar.text_input('Column with drug name (optional, leave blank if none)', value='drug_name')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)
use_llm    = st.sidebar.checkbox('Enable Gemini 2.0 Flash LLM (hardcoded key used)', value=True)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
    st.stop()

# load file robustly
try:
    if uploaded.name.lower().endswith('.csv'):
        # try utf-8, fallback to latin1
        try:
            raw_df = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            raw_df = pd.read_csv(uploaded, encoding='latin1')
    else:
        raw_df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else 0)
except Exception as e:
    st.error(f'Failed to read file: {e}')
    st.stop()

if col_name not in raw_df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(raw_df.columns)}')
    st.stop()

# configure Gemini model (object may be unhashable, we'll pass as _model)
model = configure_gemini(API_KEY) if (use_llm and API_KEY) else None
if use_llm and not API_KEY:
    st.warning("LLM enabled but API_KEY is empty — LLM columns will be empty.")

# -------------------- Processing (cached) --------------------
# NOTE: Streamlit caching can't hash the model object. Prefix with underscore to avoid hashing issue.
@st.cache_data
def process_df(_model, df: pd.DataFrame, text_col: str, drug_col_name: str):
    rows = []
    for _, row in df.iterrows():
        text_raw = row.get(text_col, '') or ''
        text = normalize_text(str(text_raw))
        # extract regex-based sentences and values
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')
        # keep only values <7 OR from-to deltas (from-to stored as reduction_pp)
        def allowed_hba(m):
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                return float(abs(rp)) < 7.0 or 'from-to' in (m.get('type') or '')
            for v in (m.get('values') or []):
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)) and abs(float(v)) < 7.0:
                        return True
                except:
                    pass
            return False
        hba_matches = [m for m in hba_matches if allowed_hba(m)]
        hba_extracted = [fmt_pct(m.get('reduction_pp')) if 'from-to' in (m.get('type') or '') else m.get('raw') for m in hba_matches]

        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')
        wt_extracted = [fmt_pct(m.get('reduction_pp')) if 'from-to' in (m.get('type') or '') else m.get('raw') for m in wt_matches]

        # Build sentence strings (watch: user asked specific long-sentence behavior)
        sentence_str = ' | '.join(hba_sentences) if hba_sentences else ''
        weight_sentence_str = ' | '.join(wt_sentences) if wt_sentences else ''

        drug_val = row.get(drug_col_name, '') if drug_col_name and drug_col_name in df.columns else ''

        # LLM step: read sentence column only
        llm_hba_extracted, hba_selected = ([], "")
        if _model is not None and sentence_str:
            llm_hba_extracted, hba_selected = llm_extract_from_sentence(_model, "HbA1c", sentence_str, str(drug_val))

        llm_wt_extracted, wt_selected = ([], "")
        if _model is not None and weight_sentence_str:
            llm_wt_extracted, wt_selected = llm_extract_from_sentence(_model, "Body weight", weight_sentence_str, str(drug_val))

        # ensure: if LLM extracted empty then selected must be empty
        if not llm_hba_extracted:
            hba_selected = ""
        if not llm_wt_extracted:
            wt_selected = ""

        # convert negative selected to positive numeric percent (keep formatting)
        if isinstance(hba_selected, str) and hba_selected.startswith('-'):
            hba_selected = hba_selected.replace('-', '')
            hba_selected = _norm_percent(hba_selected)
        if isinstance(wt_selected, str) and wt_selected.startswith('-'):
            wt_selected = wt_selected.replace('-', '')
            wt_selected = _norm_percent(wt_selected)

        # compute scores
        a1c_score = a1c_score_from_percent(hba_selected) if hba_selected else None
        wt_score  = weight_score_from_percent(wt_selected) if wt_selected else None

        new = row.to_dict()
        # add columns
        new.update({
            'sentence': sentence_str,
            'extracted_matches': hba_extracted,
            'LLM extracted': llm_hba_extracted,
            'selected %': hba_selected,
            'A1c Score': a1c_score,
            'weight_sentence': weight_sentence_str,
            'weight_extracted_matches': wt_extracted,
            'Weight LLM extracted': llm_wt_extracted,
            'Weight selected %': wt_selected,
            'Weight Score': wt_score,
        })
        rows.append(new)
    out = pd.DataFrame(rows)

    # keep row if either hba or weight matches
    def _list_has_items(x): return isinstance(x, list) and len(x) > 0
    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(_list_has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(_list_has_items))
    mask_keep = mask_hba | mask_wt
    out = out[mask_keep].reset_index(drop=True)

    # add counts
    out.attrs['counts'] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum()),
    )
    return out

# Run processing
out_df = process_df(model, raw_df, col_name, drug_col)

# Place LLM columns beside regex columns
def insert_after(cols, after, names):
    if after not in cols: return cols
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
display_df = display_df[cols]

if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

st.write("### Results (first 200 rows shown)")
st.dataframe(display_df.head(200))

# counts caption
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
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

st.download_button(
    'Download results as Excel',
    data=to_excel_bytes(display_df),
    file_name='results_with_llm_and_scores.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
