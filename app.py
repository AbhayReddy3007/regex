# streamlit_hba1c_weight_llm.py
import re
import math
import json
import unicodedata
from io import BytesIO

import pandas as pd
import streamlit as st

# -------------------- HARD-CODE YOUR GEMINI API KEY HERE --------------------
# Replace the empty string below with your Gemini API key (local only; don't commit).
API_KEY = ""
# ---------------------------------------------------------------------------

# Optional/lazy import of Gemini library
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor (with Gemini)", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash")

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

# Terms
re_hba1c  = re.compile(r'\bhb\s*a1c\b|\bhba1c\b|\ba1c\b', FLAGS)
re_weight = re.compile(r'\b(body\s*weight|weight|bw)\b', FLAGS)
re_reduction_cue = re.compile(
    r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b',
    FLAGS
)

# -------------------- Utils --------------------
def parse_number(s: str) -> float:
    if s is None:
        return float('nan')
    s = str(s).replace(',', '.').replace('·', '.').strip()
    try:
        return float(s)
    except Exception:
        return float('nan')

def clean_mojibake(text: str) -> str:
    if not isinstance(text, str):
        return text
    # common mojibake/encoding artifacts -> normalize
    replacements = {
        'Â·': '·',
        'â€¢': '·',
        'â‰¥': '≥',
        'âˆ’': '−',
        'Ã—': '×',
        '\ufffd': '',  # replacement char
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # normalize unicode (decompose + recompose)
    text = unicodedata.normalize('NFC', text)
    return text

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

def window_prev_next_spaces_inclusive_tokens(s: str, pos: int, n_prev_spaces: int = 5, n_next_spaces: int = 5):
    space_like = set([' ', '\t', '\n', '\r'])
    L = len(s)
    # left
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
    # right
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

# -------------------- Regex extraction core --------------------
def add_match(out, si, abs_start, m, typ, values, reduction):
    out.append({
        'raw': m.group(0),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (abs_start + m.start(), abs_start + m.end()),
    })

def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []
    # whole-sentence from->to
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red)
    # ±5 spaces window
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
    # weight fallback: previous 60 chars
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
    # dedupe
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
    # dedupe globally
    seen, filtered = set(), []
    for mm in matches:
        key = (mm['sentence_index'], mm['span'])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(mm)
    filtered.sort(key=lambda x: (x['sentence_index'], x['span'][0]))
    return filtered, sentences_used

# -------------------- LLM (Gemini 2.0 flash) helpers --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        # According to previous patterns used here, use GenerativeModel object:
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v

LLM_RULES = (
    "You are an extractor. Read the provided SENTENCE(S) and do ONLY the following:\n"
    "1) TARGET: either 'HbA1c' or 'Body weight' (this will be told in the prompt).\n"
    "2) Extract ONLY percentage changes that pertain to the TARGET.\n"
    "3) If 'from X% to Y%' appears, prefer reporting the delta (X - Y) as percentage points. You also may keep explicit 'reduced by Z%' phrases.\n"
    "4) If multiple timepoints mentioned, prefer 12-month/T12 values over 6-month/T6 values.\n"
    "5) If multiple candidate percentages exist, return them in order of relevance and choose one as the single 'selected_percent'.\n"
    "6) Return STRICT JSON ONLY with keys: {\"extracted\": [\"1.23%\",\"0.85%\"], \"selected_percent\": \"1.23%\"}.\n"
)

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_name_context: str = ""):
    """
    Returns (list_of_percent_strings, selected_percent_string)
    """
    sentence = clean_mojibake(sentence or "")
    if model is None or not sentence.strip():
        return [], ""

    prompt = (
        f"TARGET: {target_label}\n"
        f"DRUG_NAME (if available): {drug_name_context}\n\n"
        f"{LLM_RULES}\n\n"
        f"SENTENCE(S):\n{sentence}\n\n"
        "Return JSON only.\n"
    )

    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # try to parse JSON object inside response
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s:e+1])
            extracted = [_norm_percent(x) for x in (data.get("extracted") or []) if isinstance(x, str)]
            selected = _norm_percent(data.get("selected_percent", "") or "")
            return extracted, selected
    except Exception:
        pass
    return [], ""

# -------------------- Scoring functions --------------------
def score_a1c(selected_pct_str):
    if not selected_pct_str:
        return ""
    try:
        s = selected_pct_str.replace("%", "")
        v = abs(parse_number(s))
        if v >= 2.2:
            return 5
        if 1.8 <= v <= 2.1:
            return 4
        if 1.2 <= v <= 1.7:
            return 3
        if 0.8 <= v <= 1.1:
            return 2
        return 1
    except Exception:
        return ""

def score_weight(selected_pct_str):
    if not selected_pct_str:
        return ""
    try:
        s = selected_pct_str.replace("%", "")
        v = abs(parse_number(s))
        if v >= 22:
            return 5
        if 16 <= v <= 21.9:
            return 4
        if 10 <= v <= 15.9:
            return 3
        if 5 <= v <= 9.9:
            return 2
        return 1
    except Exception:
        return ""

# -------------------- UI / File input --------------------
st.sidebar.header("Options")
uploaded   = st.sidebar.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "xls", "csv"])
sheet_name = st.sidebar.text_input("Excel sheet name (blank = first sheet)", value="")
col_name_input = st.sidebar.text_input("Text column name (leave blank to auto-detect 'abstract' or 'text')", value="")
drug_col_input = st.sidebar.text_input("Drug name column (optional - leave blank if none)", value="")
use_llm = st.sidebar.checkbox("Enable Gemini 2.0 Flash LLM extraction", value=True)
show_debug = st.sidebar.checkbox("Show debug columns (reductions_pp, reduction_types)", value=False)

if not uploaded:
    st.info("Upload your CSV or Excel file using the sidebar.")
    st.stop()

# robust read (CSV with encoding attempts; Excel choose first sheet if None)
def read_input_file(uploaded_file, sheet_name):
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".csv"):
            # try utf-8, fallback to latin1
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            # read excel; if multiple sheets returned as dict, pick one sheet
            if sheet_name:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            else:
                tmp = pd.read_excel(uploaded_file, sheet_name=None)
                if isinstance(tmp, dict):
                    first_sheet = list(tmp.keys())[0]
                    df = tmp[first_sheet]
                else:
                    df = tmp
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
    return df

df = read_input_file(uploaded, sheet_name)

# If user left text column blank, try common column names
if not col_name_input:
    candidates = [c for c in df.columns if isinstance(c, str)]
    prefer = None
    for try_name in ["abstract", "Abstract", "text", "Text", "abstracts"]:
        if try_name in candidates:
            prefer = try_name
            break
    if prefer:
        col_name = prefer
    else:
        # let user pick from a selectbox
        col_name = st.sidebar.selectbox("Choose text column", options=list(df.columns))
else:
    if col_name_input not in df.columns:
        st.error(f'Column "{col_name_input}" not found. Available columns: {list(df.columns)}')
        st.stop()
    col_name = col_name_input

drug_col = None
if drug_col_input:
    if drug_col_input in df.columns:
        drug_col = drug_col_input
    else:
        st.error(f'Drug column "{drug_col_input}" not found. Available columns: {list(df.columns)}')
        st.stop()

st.success(f"Loaded {len(df)} rows. Using text column: '{col_name}'" + (f" and drug column: '{drug_col}'" if drug_col else ""))

# -------------------- Process rows with regex to create sentence columns --------------------
@st.cache_data
def process_df_regex_only(df, text_col):
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, "")
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        text = clean_mojibake(text)

        # HbA1c
        hba_matches, hba_sentences = extract_sentences(text, re_hba1c, 'hba1c')
        # allowed strict filter
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

        # Weight
        wt_matches, wt_sentences = extract_sentences(text, re_weight, 'weight')
        def fmt_extracted_wt(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        new = row.to_dict()
        new.update({
            'sentence': ' | '.join(hba_sentences) if hba_sentences else '',
            'extracted_matches': [fmt_extracted_hba(m) for m in hba_matches],
            'reductions_pp': [m.get('reduction_pp') for m in hba_matches],
            'reduction_types': [m.get('type') for m in hba_matches],
            'weight_sentence': ' | '.join(wt_sentences) if wt_sentences else '',
            'weight_extracted_matches': [fmt_extracted_wt(m) for m in wt_matches],
            'weight_reductions_pp': [m.get('reduction_pp') for m in wt_matches],
            'weight_reduction_types': [m.get('type') for m in wt_matches],
        })
        rows.append(new)
    out = pd.DataFrame(rows)
    # keep rows if either qualifies
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

out_df = process_df_regex_only(df, col_name)

# -------------------- Configure Gemini model --------------------
model = configure_gemini(API_KEY) if use_llm else None
if use_llm and (not API_KEY):
    st.warning("LLM enabled but API_KEY is empty — please edit the script to hardcode your key or uncheck LLM.")

# -------------------- Run LLM over sentence columns --------------------
def safe_llm_extract(target_label, sentence, drug_context):
    try:
        ex, sel = llm_extract_from_sentence(model, target_label, sentence, drug_context)
        # normalize and only keep percent-like strings
        ex = [x for x in (ex or []) if re.search(r'\d', x)]
        sel = (sel or "")
        return ex, sel
    except Exception:
        return [], ""

if model is None:
    # fill empty lists if LLM not available
    out_df["LLM extracted"] = [[] for _ in range(len(out_df))]
    out_df["selected %"] = ["" for _ in range(len(out_df))]
    out_df["Weight LLM extracted"] = [[] for _ in range(len(out_df))]
    out_df["Weight selected %"] = ["" for _ in range(len(out_df))]
else:
    l_extracted = []
    l_selected = []
    w_extracted = []
    w_selected = []
    for idx, row in out_df.iterrows():
        sent = row.get("sentence", "") or ""
        wsent = row.get("weight_sentence", "") or ""
        drug_ctx = ""
        if drug_col and drug_col in df.columns:
            # try fetch corresponding drug (original df index might not match if we filtered out rows;
            # attempt to find by row's index value)
            # We rely on original df having a unique index; safer: check if 'drug_name' present in row dict
            drug_ctx = row.get(drug_col, "") or ""
        ex, sel = safe_llm_extract("HbA1c", sent, drug_ctx)
        wex, wsel = safe_llm_extract("Body weight", wsent, drug_ctx)
        l_extracted.append(ex)
        l_selected.append(sel)
        w_extracted.append(wex)
        w_selected.append(wsel)
    out_df["LLM extracted"] = l_extracted
    out_df["selected %"] = l_selected
    out_df["Weight LLM extracted"] = w_extracted
    out_df["Weight selected %"] = w_selected

# Post-process: normalize selected % to positive numbers and compute scores
def normalize_selected_percent(s):
    if not s: return ""
    s = s.replace("%", "").replace(" ", "")
    try:
        v = parse_number(s)
        if math.isnan(v):
            return ""
        v = abs(v)
        return fmt_pct(v)
    except Exception:
        return ""

out_df["selected %"] = out_df["selected %"].astype(str).apply(_norm_percent).apply(normalize_selected_percent)
out_df["Weight selected %"] = out_df["Weight selected %"].astype(str).apply(_norm_percent).apply(normalize_selected_percent)

out_df["A1c Score"] = out_df["selected %"].apply(score_a1c)
out_df["Weight Score"] = out_df["Weight selected %"].apply(score_weight)

# -------------------- Cleanup: remove mojibake in text columns to avoid weird characters in Excel --------------------
text_cols = [c for c in out_df.columns if out_df[c].dtype == object]
for c in text_cols:
    out_df[c] = out_df[c].apply(lambda x: clean_mojibake(str(x)) if pd.notna(x) else x)

# -------------------- Reorder columns to put LLM columns beside regex columns --------------------
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
display_df = display_df[cols]

# -------------------- Show results --------------------
st.write("### Results (first 200 rows shown)")
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])
st.dataframe(display_df.head(200))

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
    buf = BytesIO()
    # ensure utf-8-friendly Excel
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.getvalue()

st.download_button(
    "Download results as Excel",
    data=to_excel_bytes(display_df),
    file_name="results_with_llm_and_scores.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# -------------------- End --------------------
