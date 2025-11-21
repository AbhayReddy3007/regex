# streamlit_hba1c_weight_strict_llm_full.py
# Full Streamlit app — regex extraction + Gemini LLM (hardcode API_KEY)
# Requirements:
#   pip install streamlit pandas openpyxl google-generativeai
# Run:
#   streamlit run streamlit_hba1c_weight_strict_llm_full.py

import re
import math
import json
from io import BytesIO
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

# -------------------- HARD-CODE YOUR GEMINI API KEY HERE --------------------
API_KEY = ""  # <-- PUT your Gemini API KEY string here
# ---------------------------------------------------------------------------

# attempt lazy import of google.generativeai
GENAI_AVAILABLE = False
genai = None
try:
    import google.generativeai as _genai
    genai = _genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Extractor (regex + LLM)", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex extraction + optional Gemini LLM")

# -------------------- Sidebar / Options --------------------
st.sidebar.header("Upload & options")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "xls", "csv"])
text_col = st.sidebar.text_input("Column that contains abstracts/text", value="abstract")
drug_col = st.sidebar.text_input("Column with drug name (optional — used to disambiguate)", value="drug_name")
sheet_name = st.sidebar.text_input("Excel sheet name (blank = first sheet)", value="")
use_llm = st.sidebar.checkbox("Enable Gemini LLM (use API_KEY above)", value=False)
show_debug = st.sidebar.checkbox("Show debug columns", value=False)
baseline_weight_kg = st.sidebar.number_input("Baseline weight (kg) for kg->% fallback", value=70.0, min_value=1.0, step=0.1)

if not uploaded:
    st.info('Upload your CSV/XLSX file and pick the text column (default "abstract").')
    st.stop()

# -------------------- Regex building blocks --------------------
NUM = r'(?:[+-]?\d+(?:[.,·]\d+)?)'
PCT = rf'({NUM})\s*%'
DASH = r'(?:-|–|—)'

FROM_TO = rf'from\s+({NUM})\s*%\s*(?:to|->|{DASH})\s*({NUM})\s*%'
FROM_TO_KG = rf'from\s+({NUM})\s*(?:kg|kgs|kilogram|kilograms)\s*(?:to|->|{DASH})\s*({NUM})\s*(?:kg|kgs|kilogram|kilograms)?'
REDUCE_BY = rf'(?:reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing)|improv(?:ed|ement))\s*(?:by\s*)?({NUM})\s*%'
ABS_PP = rf'(?:absolute\s+reduction\s+of|reduction\s+of|percentage\s+points|pp\b)\s*({NUM})(?:\s*(?:percentage\s*points|pp)?)?'
RANGE_PCT = rf'({NUM})\s*{DASH}\s*({NUM})\s*%'

FLAGS = re.IGNORECASE
re_pct = re.compile(PCT, FLAGS)
re_fromto = re.compile(FROM_TO, FLAGS)
re_fromto_kg = re.compile(FROM_TO_KG, FLAGS)
re_reduce_by = re.compile(REDUCE_BY, FLAGS)
re_abs_pp = re.compile(ABS_PP, FLAGS)
re_range = re.compile(RANGE_PCT, FLAGS)

re_hba1c = re.compile(r'\bhb\s*a1c\b|\bhba1c\b|\ba1c\b', FLAGS)
re_weight = re.compile(r'\b(body\s*weight|weight|bw)\b', FLAGS)
re_reduction_cue = re.compile(r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing)|improv(?:ed|ement))\b', FLAGS)
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

def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(?<=[\.!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = f"{float(v):.3f}".rstrip('0').rstrip('.')
    return f"{s}%"

def norm_percent_token(v) -> str:
    """Normalize token or numeric to absolute percent string or empty."""
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        try:
            return fmt_pct(abs(float(v)))
        except:
            return ""
    s = str(v).strip().replace(' ', '').replace('·', '.').replace(',', '.')
    s = s.rstrip('%')
    # remove trailing 'pp'
    s = re.sub(r'(?i)pp$', '', s)
    if not re.match(r'^[+-]?\d+(?:\.\d+)?$', s):
        return ""
    try:
        return fmt_pct(abs(float(s)))
    except:
        return ""

# inclusive ±5-space window helper
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

# add match helper
def add_match(out, si, abs_start, m, typ, values, reduction):
    out.append({
        'raw': m.group(0),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (abs_start + m.start(), abs_start + m.end()),
    })

# -------------------- Core extraction functions (regex) --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []

    # from->to percent (percentage tokens)
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        # for percent from->to: compute percent-point difference or relative percent depending on tag
        red = None
        if not math.isnan(a) and not math.isnan(b):
            # For HbA1c we might prefer percentage point diff (a-b) — user earlier wanted difference percent,
            # but many earlier iterations requested: for from x% to y% use difference (x-y) OR relative ((x-y)/x*100).
            # We'll keep authorship-simple: store absolute difference (a - b) as reduction_pp and also allow callers to compute relative if needed.
            red = a - b
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to', [a, b], red)

    # from kg to kg
    for m in re_fromto_kg.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red = None
        if not math.isnan(a) and not math.isnan(b) and a != 0:
            # compute percent change from kg -> percent
            red = ((a - b) / a) * 100.0
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to-kg', [a, b], red)

    any_window_hit = False
    for hh in term_re.finditer(sent):
        seg, abs_s, _ = window_prev_next_spaces_inclusive_tokens(sent, hh.end(), 5, 5)

        for m in re_reduce_by.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:reduce_by', [v], v)

        for m in re_abs_pp.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:abs_pp', [v], v)

        for m in re_range.finditer(seg):
            any_window_hit = True
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            rep = None
            if not math.isnan(a) and not math.isnan(b):
                rep = max(a, b)
            add_match(matches, si, abs_s, m, f'{tag_prefix}:range', [a, b], rep)

        for m in re_pct.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:pct', [v], v)

    # weight fallback: previous % within 60 chars if no window hit
    if tag_prefix == 'weight' and not any_window_hit:
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
                add_match(matches, si, abs_start, last_pct, f'{tag_prefix}:pct_prev60', [v], v)

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
    matches = []
    sentences_used = []
    sents = split_sentences(text)
    for si, sent in enumerate(sents):
        # original strict criterion: require term + pct + reduction cue
        # Relax a bit: accept if term exists AND (> pct OR from->to OR abs_pp OR range OR reduction cue)
        has_term = bool(term_re.search(sent))
        has_pct = bool(re_pct.search(sent))
        has_fromto = bool(re_fromto.search(sent)) or bool(re_fromto_kg.search(sent))
        has_abs = bool(re_abs_pp.search(sent))
        has_range = bool(re_range.search(sent))
        has_cue = bool(re_reduction_cue.search(sent))

        keep = False
        if has_term and (has_pct or has_fromto or has_abs or has_range or has_cue):
            keep = True
        # additionally, if from->to exists in adjacent sentence and this term is in neighbor, include both
        if not keep and has_fromto:
            prev_has = (si > 0) and bool(term_re.search(sents[si-1]))
            next_has = (si+1 < len(sents)) and bool(term_re.search(sents[si+1]))
            if prev_has or next_has:
                keep = True

        if not keep:
            continue
        sentences_used.append(sent)
        matches.extend(extract_in_sentence(sent, si, term_re, tag_prefix))

    # dedupe again
    seen = set()
    filtered = []
    for mm in matches:
        key = (mm['sentence_index'], mm['span'])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(mm)
    filtered.sort(key=lambda x: (x['sentence_index'], x['span'][0]))
    return filtered, sentences_used

# -------------------- Gemini configuration and safe call wrapper --------------------
def configure_gemini(api_key: str) -> Tuple[Optional[object], str]:
    """Return (model_handle_or_none, configure_error_str)."""
    if not api_key:
        return None, "API_KEY empty"
    if not GENAI_AVAILABLE:
        return None, "google.generativeai not installed/importable"
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        # still attempt to construct model handle if available
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            return model, f"genai.configure raised: {repr(e)} (model object constructed)"
        except Exception:
            return None, f"genai.configure raised: {repr(e)}"
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model, ""
    except Exception as e:
        return None, f"Failed to instantiate model: {repr(e)}"

# safe call: try likely SDK patterns deterministically
def call_model_deterministic(model_handle, prompt: str, temperature: float = 0.0) -> Tuple[str, str]:
    last_err = ""
    if model_handle is None:
        return "", "model_handle is None"
    try:
        # preferred: model_handle.generate_content
        if hasattr(model_handle, "generate_content"):
            resp = model_handle.generate_content(prompt, temperature=temperature)
            text = getattr(resp, "text", None) or getattr(resp, "content", None) or ""
            if text:
                return str(text).strip(), ""
    except Exception as e:
        last_err = f"generate_content error: {repr(e)}"

    # fallback to genai.* top-level helpers if available
    try:
        if genai is not None:
            # genai.generate_text
            if hasattr(genai, "generate_text"):
                resp = genai.generate_text(model="gemini-2.0-flash", prompt=prompt, temperature=temperature)
                if hasattr(resp, "text"):
                    return getattr(resp, "text") or "", ""
                # dict-like handling
                if isinstance(resp, dict):
                    text = resp.get("candidates", [{}])[0].get("content") or resp.get("output") or ""
                    return text, ""
            # genai.generate
            if hasattr(genai, "generate"):
                resp = genai.generate(model="gemini-2.0-flash", prompt=prompt, temperature=temperature)
                # Try to extract text fields robustly
                if isinstance(resp, dict):
                    text = resp.get("candidates", [{}])[0].get("content") or resp.get("output") or resp.get("text", "")
                    return text, ""
                text = getattr(resp, "text", None) or str(resp)
                return text, ""
    except Exception as e:
        last_err = (last_err + " ; genai generate error: " + repr(e)).strip()

    return "", last_err or "No supported call pattern succeeded"

# -------------------- LLM prompt & wrapper --------------------
LLM_RULES = (
    "You are a strict extractor. Read SENTENCE and extract ONLY percentages (or compute from from->to) for the TARGET.\n"
    "Return EXACTLY and ONLY one JSON object with keys: {\"extracted\": [...], \"selected_percent\": \"...%\"}.\n"
    "- 'extracted' should be an array of percent strings like '1.23%'.\n"
    "- 'selected_percent' should be the single best percent chosen by the LLM or \"\" if none.\n"
    "- Prefer percent tokens tied to the TARGET. If multiple timepoints appear prefer 12 months / T12 over 6 months / T6.\n"
    "- For 'from X% to Y%' give difference X - Y as percent (e.g. from 8% to 6% -> 2%).\n"
    "- If sentence mentions drug and DRUG_HINT is provided, prefer values which belong to that drug.\n"
)

def llm_extract_from_sentence(model_handle, target_label: str, sentence: str, drug_hint: str = "") -> Tuple[List[str], str, str]:
    """
    Call LLM to extract percents from sentence. Return (extracted_list, selected_percent, debug_json).
    If parsing fails, the function will try to extract percentages from the LLM text with regex as fallback.
    """
    debug = {"model_text": "", "model_error": "", "allowed_candidates": []}

    if model_handle is None:
        return [], "", json.dumps({"error": "model not configured"})

    # Build prompt
    allowed_candidates = []  # we'll let LLM infer — but pass some guidance
    prompt = "\n".join([
        f"TARGET: {target_label}",
        f"DRUG_HINT: {drug_hint}" if drug_hint else "",
        LLM_RULES,
        "SENTENCE:",
        sentence,
        "Return JSON only."
    ])
    text, err = call_model_deterministic(model_handle, prompt, temperature=0.0)
    debug["model_text"] = text or ""
    debug["model_error"] = err or ""
    try:
        if not text:
            return [], "", json.dumps(debug)
        # Try parsing JSON block in model output
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            js = text[start:end+1]
            parsed = json.loads(js)
            extracted_raw = parsed.get("extracted", []) or []
            selected_raw = parsed.get("selected_percent", "") or ""
            # normalize
            extracted = []
            for tok in extracted_raw:
                n = norm_percent_token(tok)
                if n:
                    extracted.append(n)
            selected = norm_percent_token(selected_raw)
            # ensure selected is one of extracted — if not, keep selected only if non-empty
            if selected and extracted and selected not in extracted:
                # allow selected even if not in extracted (user request allowed LLM to propose cleaned value)
                pass
            return extracted, selected, json.dumps(debug)
        else:
            # fallback: try to find percents in model text via regex
            found = []
            for m in re_pct.finditer(text):
                n = norm_percent_token(m.group(1))
                if n and n not in found:
                    found.append(n)
            sel = found[0] if found else ""
            return found, sel, json.dumps(debug)
    except Exception as e:
        debug["model_error"] = debug.get("model_error", "") + " ; parse error: " + repr(e)
        # fallback to regex on model output
        found = []
        for m in re_pct.finditer(text or ""):
            n = norm_percent_token(m.group(1))
            if n and n not in found:
                found.append(n)
        sel = found[0] if found else ""
        return found, sel, json.dumps(debug)

# -------------------- Scoring functions --------------------
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

# -------------------- Processing pipeline (cached). Note: first arg name starts with underscore to avoid streamlit hash error ----------
@st.cache_data
def process_df(_model_handle, df_in: pd.DataFrame, text_col: str, drug_col_name: str, baseline_kg: float, use_llm_flag: bool):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, '')
        if not isinstance(text_orig, str):
            text_orig = '' if pd.isna(text_orig) else str(text_orig)

        # extract regex-based sentences & matches
        hba_matches, hba_sentences = extract_sentences(text_orig, re_hba1c, 'hba1c')
        wt_matches, wt_sentences = extract_sentences(text_orig, re_weight, 'weight')

        def fmt_extracted(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                # use reduction_pp if present else raw
                rp = m.get('reduction_pp')
                return fmt_pct(rp) if rp is not None else m.get('raw', '')
            else:
                vals = m.get('values') or []
                if vals:
                    try:
                        return fmt_pct(vals[0])
                    except:
                        return m.get('raw', '')
                return m.get('raw', '')

        hba_regex_vals = [fmt_extracted(m) for m in hba_matches]
        wt_regex_vals = [fmt_extracted(m) for m in wt_matches]

        sentence_str = ' | '.join(hba_sentences) if hba_sentences else ''
        weight_sentence_str = ' | '.join(wt_sentences) if wt_sentences else ''

        drug_hint = ""
        if drug_col_name and drug_col_name in df_in.columns:
            drug_hint = str(row.get(drug_col_name, '') or "")

        # ---------- LLM extraction ----------
        hba_llm_extracted = []
        hba_selected = ""
        hba_debug = ""
        if use_llm_flag and _model_handle is not None and sentence_str:
            hba_llm_extracted, hba_selected, hba_debug = llm_extract_from_sentence(_model_handle, "HbA1c", sentence_str, drug_hint)
        else:
            hba_debug = json.dumps({"model_error": "model disabled or not configured"})

        # If LLM extracted is empty, selected must be empty (user requirement)
        if not hba_llm_extracted:
            hba_selected = ""

        # Weight LLM extraction
        wt_llm_extracted = []
        wt_selected = ""
        wt_debug = ""
        if use_llm_flag and _model_handle is not None and weight_sentence_str:
            wt_llm_extracted, wt_selected, wt_debug = llm_extract_from_sentence(_model_handle, "Body weight", weight_sentence_str, drug_hint)
        else:
            wt_debug = json.dumps({"model_error": "model disabled or not configured"})

        if not wt_llm_extracted:
            wt_selected = ""

        # Normalize selecteds (positive absolute)
        if hba_selected:
            hba_selected = norm_percent_token(hba_selected)
        if wt_selected:
            wt_selected = norm_percent_token(wt_selected)

        # Scores
        a1c_score = compute_a1c_score(hba_selected)
        weight_score = compute_weight_score(wt_selected)

        # assemble weight kg tokens for debug
        kg_tokens = []
        for m in re_kg.finditer(weight_sentence_str):
            v = parse_number(m.group(1))
            if not math.isnan(v):
                kg_tokens.append(f"{v} kg")

        new = row.to_dict()
        new.update({
            "sentence": sentence_str,
            "extracted_matches": hba_regex_vals,
            "LLM extracted": hba_llm_extracted,
            "selected %": hba_selected,
            "A1c Score": a1c_score,
            "weight_sentence": weight_sentence_str,
            "weight_extracted_matches": wt_regex_vals,
            "Weight LLM extracted": wt_llm_extracted,
            "weight KG": kg_tokens,
            "Weight selected %": wt_selected,
            "Weight Score": weight_score,
            "hba_debug": hba_debug,
            "wt_debug": wt_debug
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # keep rows where either HbA1c OR weight has extracted regex matches
    def has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask_hba = (out["sentence"].astype(str).str.len() > 0) & (out["extracted_matches"].apply(has_items))
    mask_wt = (out["weight_sentence"].astype(str).str.len() > 0) & (out["weight_extracted_matches"].apply(has_items))
    mask_keep = mask_hba | mask_wt
    out = out[mask_keep].reset_index(drop=True)

    out.attrs["counts"] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum())
    )
    return out

# -------------------- Read uploaded file robustly --------------------
try:
    if uploaded.name.lower().endswith('.csv'):
        # try UTF-8 first; fallback to latin-1
        try:
            df = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding='latin-1', engine='python')
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name if sheet_name else None)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

if text_col not in df.columns:
    st.error(f'Column "{text_col}" not found. Available columns: {list(df.columns)}')
    st.stop()

if drug_col and drug_col not in df.columns:
    st.sidebar.warning(f'Drug column "{drug_col}" not found — drug-based disambiguation will be skipped.')

# configure LLM if requested
_model_handle = None
configure_error = ""
if use_llm:
    _model_handle, configure_error = configure_gemini(API_KEY)
    if configure_error:
        st.sidebar.warning(f"Gemini configure note: {configure_error}")

st.success(f"Loaded {len(df)} rows — processing...")

# -------------------- Run processing --------------------
out_df = process_df(_model_handle, df, text_col, drug_col, baseline_weight_kg, use_llm)

# reorder to place LLM columns next to regex columns
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
# optionally hide debug columns
if not show_debug:
    for c in ["hba_debug", "wt_debug"]:
        if c in cols:
            cols.remove(c)
display_df = display_df[[c for c in cols if c in display_df.columns]]

# -------------------- Show results --------------------
st.write("### Results (first 200 rows shown)")
st.dataframe(display_df.head(200))

counts = out_df.attrs.get("counts", None)
if counts:
    st.caption(f"Kept {counts['kept']} of {counts['total']} rows — HbA1c-only: {counts['hba_only']} Weight-only: {counts['wt_only']} Both: {counts['both']}")

# Download
@st.cache_data
def to_excel_bytes(df_obj):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_obj.to_excel(writer, index=False)
    buf.seek(0)
    return buf.getvalue()

st.download_button("Download results as Excel", data=to_excel_bytes(display_df), file_name="results_hba1c_weight_llm.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
