# streamlit_hba1c_weight_llm_final.py
import os
import re
import math
import json
import traceback
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== CONFIG =====================
API_KEY = ""   # optional; leave blank if using GCP service-account JSON
BASELINE_WEIGHT = 105.0  # kg
BASELINE_A1C = 8.2       # default baseline A1c when X is missing

# Attempt to import Gemini client (optional)
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor (final)", layout="wide")
st.title("HbA1c / A1c + Body Weight — regex + Gemini 2.0 Flash (LLM-selected only)")

# -------------------- small helpers --------------------
def parse_number(s: str) -> float:
    if s is None:
        return float('nan')
    s = str(s).replace(',', '.').replace('·', '.').strip()
    try:
        return float(s)
    except Exception:
        return float('nan')

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f"{float(v):.2f}".rstrip('0').rstrip('.')
    return f"{s}%"

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if not v:
        return ""
    if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
        v = v.replace(",", ".").replace("·", ".") + "%"
    if v.endswith("%"):
        num = v[:-1].replace(",", ".").replace("·", ".")
        try:
            f = float(num)
            s = f"{f:.2f}".rstrip("0").rstrip(".")
            return s + "%"
        except Exception:
            return v
    return v

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
re_weight_unit = re.compile(r'\bkg\b|\bkilograms?\b', FLAGS)

GROUP_RE = re.compile(r'\b(?:group|arm|cohort|treatment)\s*[:\-]?\s*([A-Za-z0-9_\-]+)|\b([A-Za-z0-9_\-]+)\s+(?:group|arm)\b', re.I)
RE_MG = re.compile(r'([+-]?\d+(?:[.,]\d+)?)\s*mg', re.I)
RE_MONTHS = re.compile(r'\b(T\d{1,2}|\d{1,2}\s*(?:months|mos|mo|m))\b', re.I)
RE_KG = re.compile(r'([+-]?\d+(?:[.,·]\d+)?)\s*(?:kg|kilograms?)', re.I)

# -------------------- Utilities --------------------
def split_sentences(text: str):
    if not isinstance(text, str):
        return []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def sentence_meets_criterion(sent: str, term_re: re.Pattern) -> bool:
    has_term = bool(term_re.search(sent))
    has_pct = bool(re_pct.search(sent))
    has_kg = bool(re_weight_unit.search(sent))
    has_cue = bool(re_reduction_cue.search(sent))
    if term_re == re_weight:
        return has_term and (has_pct or has_kg or re_fromto.search(sent)) and has_cue
    return has_term and (has_pct or re_fromto.search(sent)) and has_cue

def extract_durations(text: str) -> str:
    DURATION_RE = re.compile(
        r'\b(?:T\d{1,2}|'                                       # T6, T12
        r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:weeks?|wks?|wk|w)\b|'   # 12 weeks, 6-12 weeks
        r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:months?|mos?|mo)\b|'    # 6 months, 12-mo, 6-12 months
        r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:days?|d)\b|'            # days
        r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:years?|yrs?)\b|'        # years
        r'\d{1,3}-week\b|\d{1,3}-month\b|\d{1,3}-mo\b)',         # hyphenated forms
        FLAGS
    )
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

def add_match(out, si, abs_start, m, typ, values, reduction, group_label=None, strength_mg=None, timepoint_mo=None):
    d = {
        'raw': m.group(0) if hasattr(m, 'group') else str(m),
        'type': typ,
        'values': values,
        'reduction_pp': reduction,
        'sentence_index': si,
        'span': (abs_start + (m.start() if hasattr(m, 'start') else 0), abs_start + (m.end() if hasattr(m, 'end') else 0)),
    }
    if group_label:
        d['group_label'] = group_label
    if strength_mg is not None:
        d['strength_mg'] = strength_mg
    if timepoint_mo is not None:
        d['timepoint_mo'] = timepoint_mo
    out.append(d)

def normalize_drug_name(name: str) -> str:
    if not name:
        return ''
    return re.sub(r'[^a-z0-9]', '', name.lower())

def tokenize_drug_name(name: str):
    if not name:
        return []
    toks = re.split(r'[^a-z0-9]+', name.lower())
    return [t for t in toks if len(t) >= 2]

def sentence_refers_to_drug(sentence: str, drug_name: str, aliases: list = None, relax_match: bool = False) -> bool:
    if not drug_name:
        return False
    s = sentence.lower()
    dn = drug_name.lower().strip()
    if re.search(r'\b' + re.escape(dn) + r'\b', s):
        return True
    if aliases:
        for a in aliases:
            if a and re.search(r'\b' + re.escape(a.lower().strip()) + r'\b', s):
                return True
    tokens = tokenize_drug_name(dn)
    if tokens:
        hits = 0
        for t in tokens:
            if t and re.search(r'\b' + re.escape(t) + r'\b', s):
                hits += 1
        if hits >= 1:
            return True
    if re.search(r'\b' + re.escape(dn) + r'\b\s*(?:group|arm|cohort|treatment)\b', s) or re.search(r'(?:group|arm)\s*(?:[:\-]?\s*)' + re.escape(dn), s):
        return True
    if relax_match:
        return True
    return False

def find_group_label(sent: str) -> str:
    m = GROUP_RE.search(sent)
    if not m:
        return ""
    for g in m.groups():
        if g:
            return g.strip().lower()
    return ""

def parse_strength(sent: str):
    m = RE_MG.search(sent)
    if m:
        return parse_number(m.group(1))
    return None

def parse_timepoint_months(sent: str):
    m = RE_MONTHS.search(sent)
    if not m:
        return None
    tok = m.group(0)
    t = re.search(r'(\d{1,2})', tok)
    if t:
        return int(t.group(0))
    return None

# -------------------- Core extraction --------------------
def extract_in_sentence(sent: str, si: int, term_re: re.Pattern, tag_prefix: str):
    matches = []
    group_label = find_group_label(sent)
    strength = parse_strength(sent)
    timepoint = parse_timepoint_months(sent)
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1)); b = parse_number(m.group(2))
        red_pp = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        rel = None
        if not (math.isnan(a) or math.isnan(b)) and a != 0:
            rel = ((a - b) / a) * 100.0
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], red_pp, group_label=group_label, strength_mg=strength, timepoint_mo=timepoint)
        if rel is not None:
            rel_raw = f"{rel:.6f}%"
            matches.append({
                'raw': rel_raw,
                'type': f'{tag_prefix}:from-to_relative_percent',
                'values': [a, b, rel],
                'reduction_pp': red_pp,
                'sentence_index': si,
                'span': (m.start(), m.end()),
                'group_label': group_label,
                'strength_mg': strength,
                'timepoint_mo': timepoint,
            })
    any_window_hit = False
    for hh in term_re.finditer(sent):
        seg, abs_s, _ = window_prev_next_spaces_inclusive_tokens(sent, hh.end(), 5, 5)
        for m in re_reduce_by.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:percent_or_pp_pmSpaces5', [v], v, group_label=group_label, strength_mg=parse_strength(seg), timepoint_mo=parse_timepoint_months(seg))
        for m in re_abs_pp.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:pp_word_pmSpaces5', [v], v, group_label=group_label, strength_mg=parse_strength(seg), timepoint_mo=parse_timepoint_months(seg))
        for m in re_range.finditer(seg):
            any_window_hit = True
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            rep = None if (math.isnan(a) or math.isnan(b)) else max(a, b)
            add_match(matches, si, abs_s, m, f'{tag_prefix}:range_percent_pmSpaces5', [a, b], rep, group_label=group_label, strength_mg=parse_strength(seg), timepoint_mo=parse_timepoint_months(seg))
        for m in re_pct.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:percent_pmSpaces5', [v], v, group_label=group_label, strength_mg=parse_strength(seg), timepoint_mo=parse_timepoint_months(seg))
        if tag_prefix == 'weight':
            for m in re.finditer(r'([+-]?\d+(?:[.,·]\d+)?)\s*(?:kg|kilograms?)', seg, FLAGS):
                any_window_hit = True
                v = parse_number(m.group(1))
                add_match(matches, si, abs_s, m, f'{tag_prefix}:kg_pmSpaces5', [v, 'kg'], v, group_label=group_label, strength_mg=parse_strength(seg), timepoint_mo=parse_timepoint_months(seg))
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
                add_match(matches, si, abs_start, last_pct, f'{tag_prefix}:percent_prev60chars', [v], v, group_label=group_label, strength_mg=parse_strength(left_chunk), timepoint_mo=parse_timepoint_months(left_chunk))
    seen = set()
    uniq = []
    for mm in matches:
        if mm['span'] in seen:
            continue
        seen.add(mm['span'])
        uniq.append(mm)
    uniq.sort(key=lambda x: x['span'][0])
    return uniq

def extract_sentences(text: str, term_re: re.Pattern, tag_prefix: str, row_drug_name: str = '', aliases: list = None, relax_drug_matching: bool = False):
    matches, sentences_used = [], []
    for si, sent in enumerate(split_sentences(text)):
        if not sentence_meets_criterion(sent, term_re):
            continue
        if row_drug_name and not sentence_refers_to_drug(sent, row_drug_name, aliases, relax_match=relax_drug_matching):
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

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
drug_col   = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns', value=False)
use_llm   = st.sidebar.checkbox('Enable Gemini LLM (Gemini 2.0 Flash)', value=True)
relax_drug_matching = st.sidebar.checkbox('Relax drug matching (allow LLM to run even if sentence lacks explicit drug token)', value=False)

st.sidebar.markdown("---")
use_gcp_json = st.sidebar.file_uploader('Upload GCP service account JSON (optional)', type=['json'])
use_gcp_json_relax = st.sidebar.checkbox('Use uploaded GCP JSON for authentication (if provided)', value=False)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
    st.stop()

try:
    if uploaded.name.lower().endswith('.csv'):
        try:
            df = pd.read_csv(uploaded, encoding='utf-8', on_bad_lines='skip')
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding='utf-8', engine='python', on_bad_lines='skip')
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

# -------------------- Gemini setup (supports GCP JSON) --------------------
def _save_uploaded_service_account(fu) -> str:
    if fu is None:
        return ""
    import tempfile
    try:
        suffix = '.json' if (hasattr(fu, 'name') and fu.name.endswith('.json')) else '.json'
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.write(fu.getvalue())
        tf.flush()
        tf.close()
        return tf.name
    except Exception:
        return ""

def configure_gemini(api_key: str, gcp_service_account_path: str = ""):
    if not GENAI_AVAILABLE:
        return None
    if gcp_service_account_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcp_service_account_path
    key = api_key or os.getenv('GENAI_API_KEY') or ""
    try:
        if key:
            genai.configure(api_key=key)
            return genai.GenerativeModel("gemini-2.0-flash")
        else:
            try:
                genai.configure()
            except TypeError:
                pass
            return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

gcp_json_path = ""
if use_gcp_json is not None and use_gcp_json_relax:
    gcp_json_path = _save_uploaded_service_account(use_gcp_json)

model = configure_gemini(API_KEY, gcp_service_account_path=gcp_json_path) if use_llm else None

# -------------------- Robust LLM wrapper --------------------
def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_hint: str = ""):
    """
    Returns: (extracted_list, selected_dict, raw_text, error_str)
    - extracted_list: list of dicts (value, type, percent, group_label, strength_mg, timepoint_mo, source_sentence)
    - selected_dict: any selected info the model returned (unused for selection here)
    - raw_text: raw model response
    - error_str: '' if ok else message
    """
    if model is None or not sentence.strip():
        return [], {}, "", "model-not-configured-or-empty-sentence"

    prompt = (
        f"TARGET: {target_label}\n"
        + (f"DRUG_HINT: {drug_hint}\n" if drug_hint else "")
        + "You MUST return exactly one JSON object only. No commentary.\n"
        + "Structure: {\"extracted\": [...], \"selected\": {...}}\n"
        + "Each extracted item: value (string, e.g., '1.23%'/ '3.5 kg'), type ('percent'|'kg'), percent (numeric if percent), group_label (optional), strength_mg (optional numeric), timepoint_mo (optional numeric), source_sentence (optional)\n"
        + "SENTENCE:\n" + sentence + "\n"
    )

    raw_text = ""
    try:
        resp = model.generate_content(prompt)
        raw_text = (getattr(resp, "text", "") or "").strip()
        m = re.search(r'\{[\s\S]*\}', raw_text)
        if not m:
            lines = raw_text.splitlines()
            joined = "\n".join(lines[:20])
            m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', joined, flags=re.DOTALL)
        if m:
            json_text = m.group(0)
            try:
                data = json.loads(json_text)
                extracted = data.get('extracted') or []
                selected = data.get('selected') or {}
                normed = []
                for x in extracted:
                    if not isinstance(x, dict):
                        continue
                    if 'value' in x and isinstance(x['value'], str) and x['value'].strip().endswith('%'):
                        try:
                            num = parse_number(x['value'].replace('%', ''))
                            if not math.isnan(num):
                                x['value'] = fmt_pct(abs(num))
                                x['percent'] = abs(num)
                        except:
                            pass
                    normed.append(x)
                return normed, selected, raw_text, ""
            except Exception as e:
                err = f"json-load-error: {e}"
        else:
            err = "no-json-block-found"
    except Exception as call_ex:
        raw_text = f"LLM call error: {call_ex}\n{traceback.format_exc()}"
        return [], {}, raw_text, f"llm-call-exception: {str(call_ex)}"

    # fallback: regex extract percents/kg (used to populate LLM extracted for visibility)
    fallback_extracted = []
    for m in re_pct.finditer(sentence):
        try:
            n = parse_number(m.group(1))
            if not math.isnan(n):
                fallback_extracted.append({'value': fmt_pct(abs(n)), 'type': 'percent', 'percent': abs(n), 'source_sentence': sentence})
        except:
            continue
    for m in RE_KG.finditer(sentence):
        try:
            n = parse_number(m.group(1))
            if not math.isnan(n):
                fallback_extracted.append({'value': f"{n} kg", 'type': 'kg', 'kg': n, 'source_sentence': sentence})
        except:
            continue
    if not fallback_extracted:
        return [], {}, raw_text, (err if 'err' in locals() else "unknown-no-fallback")
    return fallback_extracted, {}, raw_text, ("fallback-extraction-used: " + (err if 'err' in locals() else 'no-json'))

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
def process_df(_model, df_in: pd.DataFrame, text_col: str, drug_col_name: str, relax_drug_matching_flag: bool = False):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, '')
        if not isinstance(text_orig, str):
            text_orig = '' if pd.isna(text_orig) else str(text_orig)

        # NOTE: duration extraction is now deferred until AFTER selection (per requirement)
        # duration_str = extract_durations(text_orig)  <-- removed from here

        drug_hint = ''
        aliases = None
        if drug_col_name and drug_col_name in df_in.columns:
            drug_hint = str(row.get(drug_col_name, '') or '')
            aliases = [drug_hint]

        hba_matches, hba_sentences = extract_sentences(text_orig, re_hba1c, 'hba1c', drug_hint, aliases, relax_drug_matching_flag)
        wt_matches, wt_sentences   = extract_sentences(text_orig, re_weight, 'weight', drug_hint, aliases, relax_drug_matching_flag)

        def _find_relative_fromto(matches, baseline_for_missing=None):
            for m in matches:
                t = (m.get('type') or '').lower()
                if 'from-to_relative_percent' in t:
                    vals = m.get('values') or []
                    if len(vals) >= 3 and vals[2] is not None and not math.isnan(vals[2]):
                        return fmt_pct(vals[2])
            for m in matches:
                t = (m.get('type') or '').lower()
                if 'from-to_sentence' in t:
                    vals = m.get('values') or []
                    if len(vals) >= 2:
                        a = vals[0]; b = vals[1]
                        if (math.isnan(a) if a is None else False) or (math.isnan(b) if b is None else False):
                            continue
                        if (not isinstance(a, (int, float)) or math.isnan(a)) and baseline_for_missing is not None:
                            a = baseline_for_missing
                        if a is not None and b is not None and a != 0:
                            rel = ((a - b) / a) * 100.0
                            return fmt_pct(rel)
            return None

        precomputed_hba_rel = _find_relative_fromto(hba_matches, baseline_for_missing=BASELINE_A1C)
        precomputed_wt_rel  = _find_relative_fromto(wt_matches, baseline_for_missing=float(row.get('baseline_weight') or BASELINE_WEIGHT))

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
        wt_regex_vals  = [fmt_extracted(m) for m in wt_matches]

        sentence_str = ' | '.join(hba_sentences) if hba_sentences else ''
        weight_sentence_str = ' | '.join(wt_sentences) if wt_sentences else ''

        hba_llm_extracted, hba_selected, hba_llm_raw, hba_llm_err = ([], {}, "", "")
        if _model is not None and sentence_str:
            hba_llm_extracted, hba_selected, hba_llm_raw, hba_llm_err = llm_extract_from_sentence(_model, "HbA1c", sentence_str, drug_hint)

        wt_llm_extracted, wt_selected, wt_llm_raw, wt_llm_err = ([], {}, "", "")
        if _model is not None and weight_sentence_str:
            wt_llm_extracted, wt_selected, wt_llm_raw, wt_llm_err = llm_extract_from_sentence(_model, "Body weight", weight_sentence_str, drug_hint)

        # Ensure LLM-extracted lists exist (may be fallback)
        hba_llm_extracted = hba_llm_extracted or []
        wt_llm_extracted = wt_llm_extracted or []

        # Selected % must be only from LLM extracted values.
        def percent_str_to_raw(pct_str):
            if not pct_str:
                return None
            s = pct_str.replace('%', '').replace(',', '.').strip()
            try:
                return abs(float(s))
            except:
                return None

        def kg_to_percent_raw(kg_str, baseline):
            if not kg_str:
                return None
            m = re.search(r'([+-]?\d+(?:[.,·]\d+)?)', kg_str)
            if not m:
                return None
            num = parse_number(m.group(1))
            if math.isnan(num) or baseline == 0:
                return None
            pct = (abs(num) / float(baseline)) * 100.0
            return pct

        baseline_for_row = float(row.get('baseline_weight') or BASELINE_WEIGHT)

        def candidates_from_llm(items, baseline_for_row):
            out = []
            for it in items:
                if isinstance(it, dict):
                    val = it.get('value')
                    typ = it.get('type')
                    meta = it.copy()
                    if typ == 'percent' and isinstance(val, str) and val.strip().endswith('%'):
                        n = percent_str_to_raw(val)
                        if n is not None:
                            out.append({'percent': n, 'raw': fmt_pct(n), 'group': meta.get('group_label'), 'strength_mg': meta.get('strength_mg'), 'timepoint_mo': meta.get('timepoint_mo'), 'relative': meta.get('relative'), 'meta': meta})
                    elif typ == 'kg' and isinstance(val, str) and 'kg' in val:
                        p = kg_to_percent_raw(val, baseline_for_row)
                        if p is not None:
                            out.append({'percent': p, 'raw': f"{val}", 'group': meta.get('group_label'), 'strength_mg': meta.get('strength_mg'), 'timepoint_mo': meta.get('timepoint_mo'), 'relative': meta.get('relative'), 'meta': meta})
                    else:
                        if 'percent' in meta and isinstance(meta['percent'], (int, float)):
                            out.append({'percent': abs(float(meta['percent'])), 'raw': fmt_pct(abs(float(meta['percent']))), 'group': meta.get('group_label'), 'strength_mg': meta.get('strength_mg'), 'timepoint_mo': meta.get('timepoint_mo'), 'relative': meta.get('relative'), 'meta': meta})
                elif isinstance(it, str):
                    if it.strip().endswith('%'):
                        n = percent_str_to_raw(it)
                        if n is not None:
                            out.append({'percent': n, 'raw': _norm_percent(it), 'group': None, 'strength_mg': None, 'timepoint_mo': None, 'relative': None, 'meta': {}})
                    elif 'kg' in it:
                        p = kg_to_percent_raw(it, baseline_for_row)
                        if p is not None:
                            out.append({'percent': p, 'raw': it, 'group': None, 'strength_mg': None, 'timepoint_mo': None, 'relative': None, 'meta': {}})
            return out

        hba_llm_cands = candidates_from_llm(hba_llm_extracted, baseline_for_row)
        wt_llm_cands = candidates_from_llm(wt_llm_extracted, baseline_for_row)

        # tie-breaker rules
        group_wt = {}
        for item in wt_llm_cands:
            g = (item.get('group') or '').lower() or None
            p = item.get('percent')
            if g and p is not None:
                group_wt.setdefault(g, []).append(p)
        group_wt_max = {g: max(vals) for g, vals in group_wt.items() if vals}

        def choose_best(candidates):
            if not candidates:
                return None
            with_strength = [c for c in candidates if c.get('strength_mg')]
            if with_strength:
                with_strength.sort(key=lambda x: (x.get('strength_mg') or 0), reverse=True)
                return with_strength[0]
            with_time = [c for c in candidates if c.get('timepoint_mo')]
            if with_time:
                with_time.sort(key=lambda x: (x.get('timepoint_mo') or 0), reverse=True)
                return with_time[0]
            best_grp = None; best_w = -1
            for c in candidates:
                g = (c.get('group') or '').lower()
                if g and g in group_wt_max and group_wt_max[g] > best_w:
                    best_w = group_wt_max[g]
                    best_grp = c
            if best_grp:
                return best_grp
            with_rel = [c for c in candidates if c.get('relative')]
            if with_rel:
                with_rel.sort(key=lambda x: (x.get('relative') or 0), reverse=True)
                return with_rel[0]
            candidates.sort(key=lambda x: (x.get('percent') or 0), reverse=True)
            return candidates[0]

        chosen_hba = choose_best(hba_llm_cands)
        chosen_wt = choose_best(wt_llm_cands)

        # Selected % must be only from LLM extracted values:
        chosen_hba_pct = fmt_pct(chosen_hba['percent']) if chosen_hba else ''
        chosen_wt_pct = fmt_pct(chosen_wt['percent']) if chosen_wt else ''

        # Scores
        a1c_score = compute_a1c_score(chosen_hba_pct)
        weight_score = compute_weight_score(chosen_wt_pct)

        # DURATION: extract only if we have a selected % for either A1c or Weight
        if chosen_hba_pct or chosen_wt_pct:
            duration_str = extract_durations(text_orig)
        else:
            duration_str = ""

        # Prepare row output
        new = row.to_dict()
        new.update({
            'sentence': sentence_str,
            'extracted_matches': hba_regex_vals,
            'LLM extracted': hba_llm_extracted,
            'LLM_raw_response': hba_llm_raw or wt_llm_raw or "",
            'LLM_error': hba_llm_err or wt_llm_err or "",
            'selected %': chosen_hba_pct,                 # ONLY from LLM candidates
            'A1c Score': a1c_score,
            'weight_sentence': weight_sentence_str,
            'weight_extracted_matches': wt_regex_vals,
            'Weight LLM extracted': wt_llm_extracted,
            'Weight selected %': chosen_wt_pct,           # ONLY from LLM candidates
            'Weight Score': weight_score,
            'duration': duration_str,
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    def has_items(x):
        return isinstance(x, list) and len(x) > 0

    if len(out) == 0:
        out = pd.DataFrame(columns=list(df_in.columns) + ['sentence','extracted_matches','LLM extracted','LLM_raw_response','LLM_error','selected %','A1c Score','weight_sentence','weight_extracted_matches','Weight LLM extracted','Weight selected %','Weight Score','duration'])
        out.attrs['counts'] = dict(kept=0, total=0, hba_only=0, wt_only=0, both=0)
        return out

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

# -------------------- Run processing --------------------
out_df = process_df(model, df, col_name, drug_col, relax_drug_matching_flag=relax_drug_matching)

# -------------------- Prepare pretty display strings --------------------
def stringify_item(x):
    try:
        if x is None:
            return ""
        if isinstance(x, list):
            parts = []
            for it in x:
                parts.append(stringify_item(it))
            parts = [p for p in parts if p]
            return " | ".join(parts)
        if isinstance(x, dict):
            if 'value' in x and x['value']:
                return str(x['value'])
            if 'raw' in x and x['raw']:
                return str(x['raw'])
            keys = []
            for k in ('percent', 'kg', 'type', 'group_label', 'strength_mg', 'timepoint_mo'):
                if k in x and x[k] is not None and x[k] != "":
                    keys.append(f"{k}:{x[k]}")
            if keys:
                return ", ".join(keys)
            try:
                return json.dumps(x, ensure_ascii=False)
            except:
                return str(x)
        if isinstance(x, str):
            return x
        return str(x)
    except Exception:
        return str(x)

# copy for display/export
display_df = out_df.copy()
cols_to_stringify = [
    'LLM extracted',
    'Weight LLM extracted',
    'extracted_matches',
    'weight_extracted_matches',
    'LLM_raw_response',
    'LLM_error'
]
for c in cols_to_stringify:
    if c in display_df.columns:
        display_df[c] = display_df[c].apply(stringify_item)

# Keep display_df_for_view separate for rendering and export
display_df_for_view = display_df.copy()

# Reorder columns to put LLM columns near regex columns
def insert_after(cols, after, names):
    if after not in cols:
        return cols
    i = cols.index(after)
    for name in names[::-1]:
        if name in cols:
            cols.remove(name)
        cols.insert(i+1, name)
    return cols

cols = list(display_df_for_view.columns)
cols = insert_after(cols, "extracted_matches", ["LLM extracted", "LLM_raw_response", "LLM_error", "selected %", "A1c Score"])
cols = insert_after(cols, "weight_extracted_matches", ["Weight LLM extracted", "Weight selected %", "Weight Score"])
seen = set(); new_cols = []
for c in cols:
    if c not in seen:
        new_cols.append(c); seen.add(c)
display_df_for_view = display_df_for_view[new_cols]

if 'duration' in display_df_for_view.columns:
    cols_no_duration = [c for c in display_df_for_view.columns if c != 'duration']
    cols_no_duration.append('duration')
    display_df_for_view = display_df_for_view[cols_no_duration]

# -------------------- Show results --------------------
st.write("### Results (first 200 rows shown)")
if not show_debug:
    for c in ['reductions_pp', 'reduction_types', 'weight_reductions_pp', 'weight_reduction_types']:
        if c in display_df_for_view.columns:
            display_df_for_view = display_df_for_view.drop(columns=[c])

st.dataframe(display_df_for_view.head(200))

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

excel_bytes = to_excel_bytes(display_df_for_view)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_llm_from_sentence_final.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
