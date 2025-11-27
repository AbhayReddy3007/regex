# streamlit_a1c_llm_duration.py
"""
Streamlit app: HbA1c (A1c) extraction + scoring + duration column using regex + Gemini LLM.

Logic:
1. Regex finds ONLY sentences that:
   - Mention A1c / HbA1c / Hb A1c
   - Contain a reduction cue (reduce/decrease/decline/fell/...)
   - Contain a number followed by '%'
2. LLM reads ONLY those shortlisted sentences to extract A1c reduction values.
3. LLM also reads the FULL ABSTRACT to find duration.
4. Among available values, it selects the best A1c value and a score is assigned.

Extra columns:
- `shortlisted_sentences`: the sentences that passed the filters.
- `selected %`: the chosen A1c reduction.
- `A1c Score`: score based on selected %.
- `duration`: trial/timepoint duration (LLM preferred, regex fallback).

Usage:
    streamlit run streamlit_a1c_llm_duration.py
"""

import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
API_KEY = "REPLACE_WITH_YOUR_REAL_GEMINI_KEY"
# ==========================================================================

# Lazy import of Gemini so the app can still run without it
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c + Duration Extractor (LLM)", layout="wide")
st.title("HbA1c (A1c) — regex + Gemini LLM extraction + score + duration")

# -------------------- Regex helpers --------------------
NUM = r'(?:[+-]?\d+(?:[.,·]\d+)?)'
PCT = rf'({NUM})\s*%'
DASH = r'(?:-|–|—)'

FROM_TO   = rf'from\s+({NUM})\s*%\s*(?:to|->|{DASH})\s*({NUM})\s*%'
REDUCE_BY = rf'(?:reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|' \
            rf'drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\s*(?:by\s*)?({NUM})\s*%'
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
    r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|'
    r'drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b',
    FLAGS,
)

# Duration regex (fallback)
DURATION_RE = re.compile(
    r'\b(?:T\d{1,2}|'
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:weeks?|wks?|wk|w)\b|'
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:months?|mos?|mo)\b|'
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:days?|d)\b|'
    r'\d{1,3}\s*(?:-\s*\d{1,3}\s*)?(?:years?|yrs?)\b|'
    r'\d{1,3}-week\b|\d{1,3}-month\b|\d{1,3}-mo\b)',
    FLAGS,
)

# -------------------- Utilities --------------------
def parse_number(s: str) -> float:
    if s is None:
        return float("nan")
    s = str(s).replace(",", ".").replace("·", ".").strip()
    try:
        return float(s)
    except Exception:
        return float("nan")


def split_sentences(text: str):
    if not isinstance(text, str):
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = f"{float(v):.2f}".rstrip("0").rstrip(".")
    return f"{s}%"


def extract_durations_regex(text: str) -> str:
    """Fallback duration extraction directly from abstract (no LLM)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    found = []
    seen = set()
    for m in DURATION_RE.finditer(text):
        token = m.group(0).strip()
        token = re.sub(r"\s+", " ", token)
        token = token.replace("–", "-").replace("—", "-")
        token = re.sub(r"\bmos?\b", "months", token, flags=re.IGNORECASE)
        token = re.sub(r"\bmo\b", "months", token, flags=re.IGNORECASE)
        token = re.sub(r"\bwks?\b", "weeks", token, flags=re.IGNORECASE)
        token = re.sub(r"\bw\b", "weeks", token, flags=re.IGNORECASE)
        token = re.sub(r"\bd\b", "days", token, flags=re.IGNORECASE)
        token = re.sub(r"\byrs?\b", "years", token, flags=re.IGNORECASE)
        token = token.strip()
        if token.lower() not in seen:
            seen.add(token.lower())
            found.append(token)
    return " | ".join(found)


def window_prev_next_spaces_inclusive_tokens(
    s: str, pos: int, n_prev_spaces: int = 5, n_next_spaces: int = 5
):
    """Build a small window around a position using ±N spaces as boundaries."""
    space_like = {" ", "\t", "\n", "\r"}
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
        if k < L and s[k] in space_like:
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
    out.append(
        {
            "raw": m.group(0) if hasattr(m, "group") else str(m),
            "type": typ,
            "values": values,
            "reduction_pp": reduction,
            "sentence_index": si,
            "span": (
                abs_start + (m.start() if hasattr(m, "start") else 0),
                abs_start + (m.end() if hasattr(m, "end") else 0),
            ),
        }
    )

# -------------------- Core A1c extraction (regex) --------------------
def extract_in_sentence(sent: str, si: int):
    matches = []

    # 1) from X% to Y% (whole sentence)
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1))
        b = parse_number(m.group(2))
        red_pp = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        rel = None
        if not (math.isnan(a) or math.isnan(b)) and b != 0:
            rel = ((a - b) / b) * 100.0
        add_match(matches, si, 0, m, "from-to_sentence", [a, b], red_pp)
        if rel is not None:
            rel_raw = f"{rel:.6f}%"
            matches.append(
                {
                    "raw": rel_raw,
                    "type": "from-to_relative_percent",
                    "values": [a, b, rel],
                    "reduction_pp": red_pp,
                    "sentence_index": si,
                    "span": (m.start(), m.end()),
                }
            )

    # 2) ±5-space window around each A1c occurrence
    for hh in re_hba1c.finditer(sent):
        seg, abs_s, _ = window_prev_next_spaces_inclusive_tokens(
            sent, hh.end(), 5, 5
        )

        for m in re_reduce_by.finditer(seg):
            v = parse_number(m.group(1))
            add_match(
                matches,
                si,
                abs_s,
                m,
                "percent_or_pp_pmSpaces5",
                [v],
                v,
            )

        for m in re_abs_pp.finditer(seg):
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, "pp_word_pmSpaces5", [v], v)

        for m in re_range.finditer(seg):
            a = parse_number(m.group(1))
            b = parse_number(m.group(2))
            rep = None if (math.isnan(a) or math.isnan(b)) else max(a, b)
            add_match(
                matches,
                si,
                abs_s,
                m,
                "range_percent_pmSpaces5",
                [a, b],
                rep,
            )

        for m in re_pct.finditer(seg):
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, "percent_pmSpaces5", [v], v)

    # dedupe by span
    seen = set()
    uniq = []
    for mm in matches:
        if mm["span"] in seen:
            continue
        seen.add(mm["span"])
        uniq.append(mm)
    uniq.sort(key=lambda x: x["span"][0])
    return uniq


def sentence_meets_criterion(sent: str) -> bool:
    """
    Sentence must have:
      1) A1c/HbA1c mention
      2) a reduction cue
      3) a numeric % (not just >=5%, <=7%, etc.)
    """
    has_term = bool(re_hba1c.search(sent))
    has_pct = bool(re.search(r"(?<![<>≥≤])" + PCT, sent))
    has_cue = bool(re_reduction_cue.search(sent))
    return bool(has_term and has_pct and has_cue)


def extract_sentences(text: str):
    matches, sentences_used = [], []
    for si, sent in enumerate(split_sentences(text)):
        if not sentence_meets_criterion(sent):
            continue
        sentences_used.append(sent)
        matches.extend(extract_in_sentence(sent, si))

    # dedupe globally
    seen, filtered = set(), []
    for mm in matches:
        key = (mm["sentence_index"], mm["span"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(mm)
    filtered.sort(key=lambda x: (x["sentence_index"], x["span"][0]))
    return filtered, sentences_used

# -------------------- LLM rules & helpers --------------------
LLM_RULES = r"""
You are an extraction assistant.

INPUT:
- SENTENCES_FOR_A1C: a small set of sentences that each mention HbA1c/A1c, a reduction cue, and contain a percent.
- FULL_ABSTRACT_FOR_DURATION: the entire abstract text.

TASKS:
1) From SENTENCES_FOR_A1C, extract all HbA1c change magnitudes (percent reductions), and choose the best single value.
2) From FULL_ABSTRACT_FOR_DURATION, identify the trial/timepoint duration associated with the main A1c result.

Return exactly ONE JSON object and NO extra text.

Allowed keys:
{
  "extracted": ["1.75%", "2.0%"],   // array of candidate percent magnitudes (strings, may be empty)
  "selected_percent": "1.75%",      // OPTIONAL: best single percent reduction (positive, with %)
  "duration": "12 months",          // OPTIONAL: duration/timepoint (e.g., "T12", "12 months", "26 weeks")
  "confidence": 0.9                 // OPTIONAL: confidence (0.0 - 1.0)
}

Rules:
- Percent strings:
  - Must use '.' as decimal separator.
  - Must end with '%', e.g., "1.75%".
  - Represent *change magnitudes* in HbA1c (reductions), NOT thresholds or goals.
- Duration:
  - Human readable, e.g., "T12", "12 months", "26 weeks", "24-52 weeks".
  - If multiple timepoints, pick the one most clearly tied to the main A1c result (prefer 12 months/T12 over 6 months/T6 when ambiguous).
- From phrases like "from 8.0% to 6.0%":
  - You may compute relative reduction ((8 - 6) / 6 * 100) and include it as a candidate.
- Ignore:
  - thresholds like ">=5%" or "HbA1c <7.0%",
  - sample-size percentages,
  - p-values, confidence intervals, etc.
- Output must be STRICT JSON and parseable.
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


def llm_extract_a1c_and_duration(model, a1c_sentences: str, full_abstract: str):
    """
    LLM reads:
      - only qualifying sentences for A1c values
      - full abstract for duration
    Returns (extracted_list, selected_percent, duration_str).
    """
    if (
        model is None
        or not a1c_sentences
        or not a1c_sentences.strip()
        or not full_abstract
        or not full_abstract.strip()
    ):
        return [], "", ""

    prompt = (
        "SENTENCES_FOR_A1C:\n"
        + a1c_sentences
        + "\n\nFULL_ABSTRACT_FOR_DURATION:\n"
        + full_abstract
        + "\n\n"
        + LLM_RULES
        + "\nReturn JSON only."
    )

    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            data = json.loads(text[s : e + 1])
            extracted = []
            for x in (data.get("extracted") or []):
                if not isinstance(x, str):
                    continue
                x2 = _norm_percent(x)
                if re.match(r"^[+-]?\d+(?:[.,·]\d+)?%$", x2):
                    n = parse_number(x2.replace("%", ""))
                    if not math.isnan(n):
                        extracted.append(fmt_pct(abs(n)))
            selected = _norm_percent(data.get("selected_percent", "") or "")
            if selected:
                try:
                    n = parse_number(selected.replace("%", ""))
                    if not math.isnan(n):
                        selected = fmt_pct(abs(n))
                except Exception:
                    selected = ""
            duration = (data.get("duration") or "").strip()
            return extracted, selected, duration
    except Exception:
        return [], "", ""
    return [], "", ""

# -------------------- Scoring --------------------
def compute_a1c_score(selected_pct_str: str):
    """
    A1c Score:
    5: >2.2%
    4: 1.8%-2.1%
    3: 1.2%-1.7%
    2: 0.8%-1.1%
    1: <0.8%
    """
    if not selected_pct_str:
        return ""
    try:
        val = parse_number(selected_pct_str.replace("%", ""))
    except Exception:
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
def process_df(df_in: pd.DataFrame, text_col: str, model, use_llm: bool):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, "")
        if not isinstance(text_orig, str):
            text_orig = "" if pd.isna(text_orig) else str(text_orig)

        # Regex duration fallback
        duration_regex = extract_durations_regex(text_orig)

        # Find qualifying sentences & regex matches
        hba_matches, hba_sentences = extract_sentences(text_orig)
        shortlisted = " | ".join(hba_sentences) if hba_sentences else ""

        # Precomputed relative from-to if available
        def _find_precomputed_relative(matches):
            for m in matches:
                t = (m.get("type") or "").lower()
                if "from-to_relative_percent" in t or "from-to_relative" in t:
                    vals = m.get("values") or []
                    if len(vals) >= 3 and vals[2] is not None and not math.isnan(vals[2]):
                        return fmt_pct(vals[2])
            for m in matches:
                t = (m.get("type") or "").lower()
                if "from-to_sentence" in t:
                    vals = m.get("values") or []
                    if len(vals) >= 2:
                        a = vals[0]
                        b = vals[1]
                        if not (math.isnan(a) or math.isnan(b)) and b != 0:
                            rel = ((a - b) / b) * 100.0
                            return fmt_pct(rel)
            return None

        precomputed_rel = _find_precomputed_relative(hba_matches)

        # Format regex outputs
        def fmt_extracted(m):
            t = (m.get("type") or "").lower()
            if "from-to" in t:
                if "relative" in t:
                    vals = m.get("values") or []
                    if len(vals) >= 3 and vals[2] is not None and not math.isnan(vals[2]):
                        return fmt_pct(vals[2])
                    if m.get("reduction_pp") is not None and not math.isnan(m.get("reduction_pp")):
                        return fmt_pct(m.get("reduction_pp"))
                    return m.get("raw", "")
                else:
                    if m.get("reduction_pp") is not None and not math.isnan(m.get("reduction_pp")):
                        return fmt_pct(m.get("reduction_pp"))
                    return m.get("raw", "")
            if isinstance(m.get("reduction_pp"), (int, float)) and not math.isnan(
                m.get("reduction_pp")
            ):
                return fmt_pct(m.get("reduction_pp"))
            return m.get("raw", "")

        hba_regex_vals = [fmt_extracted(m) for m in hba_matches]

        # LLM extraction (if enabled, reads only shortlisted sentences + full abstract for duration)
        llm_extracted, llm_selected, llm_duration = [], "", ""
        if use_llm and model is not None and shortlisted:
            llm_extracted, llm_selected, llm_duration = llm_extract_a1c_and_duration(
                model, shortlisted, text_orig
            )

        # Final duration: prefer LLM duration; else regex
        final_duration = llm_duration if llm_duration else duration_regex

        # Selected percent: prefer LLM, else precomputed_rel, else first regex
        selected = ""
        if llm_selected:
            selected = llm_selected
        elif precomputed_rel:
            selected = precomputed_rel
        else:
            for it in hba_regex_vals:
                if isinstance(it, str) and it.strip().endswith("%"):
                    s2 = it.replace("%", "").replace(",", ".").strip()
                    try:
                        v = float(s2)
                        selected = fmt_pct(abs(v))
                        break
                    except Exception:
                        continue

        a1c_score = compute_a1c_score(selected)

        new = row.to_dict()
        new.update(
            {
                "shortlisted_sentences": shortlisted,
                "sentence": shortlisted,  # keep for debug compatibility
                "extracted_matches": hba_regex_vals,
                "LLM extracted": llm_extracted,
                "selected %": selected,
                "A1c Score": a1c_score,
                "duration": final_duration,
            }
        )
        rows.append(new)

    out = pd.DataFrame(rows)

    # Keep rows that actually have extraction
    def has_items(x):
        return isinstance(x, list) and len(x) > 0

    if not out.empty:
        mask = (out["shortlisted_sentences"].astype(str).str.len() > 0) & (
            out["extracted_matches"].apply(has_items)
            | out["LLM extracted"].apply(has_items)
        )
        out = out[mask].reset_index(drop=True)
        out.attrs["counts"] = dict(kept=int(mask.sum()), total=int(len(mask)))
    else:
        out.attrs["counts"] = dict(kept=0, total=0)

    return out

# -------------------- UI --------------------
st.sidebar.header("Options")
use_llm = st.sidebar.checkbox(
    "Enable Gemini LLM (Gemini 2.0 Flash) for A1c + duration", value=True
)
uploaded = st.sidebar.file_uploader(
    "Upload Excel (.xlsx) or CSV", type=["xlsx", "xls", "csv"]
)
col_name = st.sidebar.text_input("Column with abstracts/text", value="abstract")
show_debug = st.sidebar.checkbox(
    "Show extra debug columns (extracted_matches, LLM extracted, sentence)", value=False
)

if not uploaded:
    st.info(
        'Upload your Excel or CSV file in the left sidebar. '
        'Example: my_abstracts.xlsx with column named "abstract".'
    )
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, sheet_name=0)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

if col_name not in df.columns:
    st.error(f'Column "{col_name}" not found. Available columns: {list(df.columns)}')
    st.stop()

# Configure LLM model (global)
model = None
if use_llm:
    if not GENAI_AVAILABLE:
        st.warning("google.generativeai not available — LLM disabled.")
        use_llm = False
    else:
        key = API_KEY.strip()
        if not key:
            st.warning("No Gemini API key set (API_KEY is empty). LLM disabled.")
            use_llm = False
        else:
            model = configure_gemini(key)
            if model is None:
                st.warning("Failed to configure Gemini model. LLM disabled.")
                use_llm = False

st.success(f"Loaded {len(df)} rows. Processing...")

out_df = process_df(df, col_name, model, use_llm)

# Ensure 'duration' is the last column
if "duration" in out_df.columns:
    cols = [c for c in out_df.columns if c != "duration"]
    cols.append("duration")
    out_df = out_df[cols]

st.write("### Results (first 200 rows shown)")
display_df = out_df.copy()
if not show_debug:
    for c in ["extracted_matches", "LLM extracted", "sentence"]:
        if c in display_df.columns:
            display_df = display_df.drop(columns=[c])

st.dataframe(display_df.head(200))

counts = out_df.attrs.get("counts", None)
if counts:
    kept = counts.get("kept", 0)
    total = counts.get("total", 0)
    st.caption(f"Kept {kept} rows with A1c extraction (from {total} rows).")

# -------------------- Download --------------------
def to_excel_bytes(df_out):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(out_df)
st.download_button(
    "Download results as Excel",
    data=excel_bytes,
    file_name="a1c_results_with_llm_and_duration.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
