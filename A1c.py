# streamlit_a1c_llm_duration.py
"""
Streamlit app: HbA1c (A1c) extraction + scoring + duration column using regex + Gemini LLM.

Key rules:

1. Regex finds ONLY sentences that:
   - Mention A1c / HbA1c / Hb A1c, AND
   - Contain a reduction cue (reduce/decrease/decline/fell/...), AND
   - Contain a number with '%', AND
   - Are NOT responder-rate sentences like "proportion of subjects ... 82.4% vs 16%".

2. These shortlisted sentences are sent to the LLM.

3. LLM reads ONLY those shortlisted sentences to extract A1c reduction values
   AND reads the FULL ABSTRACT for duration.

4. LLM is instructed to:
   - Treat A1c, HbA1c, Hb A1c as same.
   - Ignore responder rates / % of subjects.
   - Ignore insulin / dose / weight / BMI changes.
   - Ignore baseline / target A1c values.
   - Ignore pure between-group "difference" values (e.g., mean difference -0.35%).

5. IMPORTANT POST-FILTER:
   Any % value produced by the LLM that does NOT literally appear as a % in the
   shortlisted sentences is discarded.
   → Relative reductions like 25% hallucinated by the LLM are dropped.
   → Relative reductions for "from X% to Y%" are handled by regex, not by LLM.

6. Drug hint:
   - Optional drug column is passed as DRUG_HINT.
   - LLM is told to only extract A1c changes clearly tied to that drug (if given).

7. selected % logic:
   - If LLM A1c reduction values list is empty → selected % is empty (no regex fallback).
   - If LLM list exists and all values are 0 → selected % = 0%.
   - Else:
       a) prefer LLM's selected_percent (if passes filtering),
       b) then regex precomputed relative from-to (if any),
       c) then first regex % value.

8. A1c Score is computed from selected %.

Columns:
- shortlisted_sentences      : sentences used for A1c extraction
- A1c reduction values       : ALL LLM-extracted A1c reduction values (joined with " | ")
- selected %                 : chosen A1c reduction per rules above
- A1c Score                  : numeric score from selected %
- duration                   : LLM duration, or regex fallback
- (debug) extracted_matches, LLM extracted, sentence
"""

import re
import math
import json
import os
from io import BytesIO
from typing import Optional

import pandas as pd
import streamlit as st

# ===================== GEMINI KEY (from environment) =====================
API_KEY = os.getenv("GEMINI_API_KEY", "")
# ========================================================================

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
REDUCE_BY = (
    rf'(?:reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|'
    rf'drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\s*(?:by\s*)?({NUM})\s*%'
)
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

# NEW: pattern to avoid responder-rate sentences
re_proportion_subjects = re.compile(
    r'\b(proportion|percentage|percent)\s+of\s+(subjects|patients|participants)\b',
    re.IGNORECASE,
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
    while j >= 0 and j < len(s) and s[j] not in space_like:
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

    # 1) from X% to Y% (whole sentence) – RELATIVE reduction from baseline X
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1))
        b = parse_number(m.group(2))
        red_pp = None if (math.isnan(a) or math.isnan(b)) else (a - b)
        rel = None
        if not (math.isnan(a) or math.isnan(b)) and a != 0:
            rel = ((a - b) / a) * 100.0  # baseline = X
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
      4) MUST NOT be a responder-rate / proportion-of-subjects sentence
    """
    has_term = bool(re_hba1c.search(sent))
    has_pct = bool(re.search(r"(?<![<>≥≤])" + PCT, sent))
    has_cue = bool(re_reduction_cue.search(sent))

    if not (has_term and has_pct and has_cue):
        return False

    # Drop sentences like "proportion of subjects/patients ... 82.4% vs 16%"
    if re_proportion_subjects.search(sent):
        return False

    return True


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

CORE IDEA:
- Treat "A1c", "HbA1c", and "Hb A1c" as exactly the same lab measurement.
- Your ONLY job is to extract **change magnitudes in A1c/HbA1c** for a specific drug/group,
  not anything else (not insulin dose, not weight, not proportions of patients, etc.).

INPUT:
- DRUG_HINT: a string with the target drug/regimen name (may be empty).
- SENTENCES_FOR_A1C: a small set of sentences that each mention A1c/HbA1c/Hb A1c and contain a percent.
- FULL_ABSTRACT_FOR_DURATION: the entire abstract text.

OUTPUT:
Return exactly ONE JSON object and NO extra text. Allowed keys:
{
  "extracted": ["1.75%", "1.35%", "25.0%"],
  "selected_percent": "25.0%",
  "duration": "6 months",
  "confidence": 0.9
}

WHAT COUNTS AS A VALID A1c REDUCTION:
You may ONLY extract a percent value if it clearly represents **how much A1c changed** over time
for a group/arm. Typical valid patterns are:

- "HbA1c decreased by 1.5%"
- "A1c reduction of 1.2%"
- "A1c fell 1.3%"
- "HbA1c changed from 8.5% to 7.0%"
- "mean change in A1c was -1.1%"

You must be able to read the text as "A1c changed by X%" (where X is your extracted value).

ABSOLUTELY EXCLUDE THE FOLLOWING (DO NOT EXTRACT FROM THEM):
1) Responder rates / proportions / percentages of patients
   - Any % tied to phrases like:
       - "proportion of subjects/patients/participants"
       - "percentage of subjects/patients/participants"
       - "percent of subjects/patients/participants"
       - "proportion achieving"
       - "percentage achieving"
       - "patients achieving HbA1c <7%"
       - "the proportion of subjects with any decrease in HbA1c was 82.4% vs 16%"
   - These are about **how many patients responded**, not how much A1c changed.

2) Between-group differences ONLY (no absolute A1c change)
   - Phrases like:
       - "difference -0.35%"
       - "mean difference -0.35%"
       - "treatment difference 0.3%"
       - "HbA1c reduction was significantly greater with drug A than with drug B (mean difference -0.35%)"
   - If the % is describing a **difference between two treatments**, and you are NOT given
     the actual change for a specific arm, you must IGNORE this % completely.

3) Insulin dose / medication dose / weight / anything that is NOT A1c
   - Ignore any % clearly tied to:
       - "insulin", "insulin dose", "insulin units"
       - "dose", "dosage", "medication dose"
       - "body weight", "weight", "kg", "BMI", "%BMIp95"
   - Example:
       "Patients with a baseline A1c of 8.0% or lower required a larger decrease in insulin
        (-22.6% vs ...)."
     → -22.6% is a change in insulin, NOT A1c.

4) Baseline/target A1c values (no change described)
   - Ignore values like:
       - "baseline A1c of 8.2%"
       - "A1c goal <7.0%"
       - "patients with A1c ≤8.0%"
   - These are baseline or target levels, not changes.
   - Only extract if the text clearly says A1c **changed by** that amount or changed **from X to Y**.

DRUG LINKING RULE (DRUG_HINT):
- If DRUG_HINT is NON-EMPTY:
  - You must only extract A1c change magnitudes that can be clearly linked to that drug/regimen.
  - Look for explicit associations such as:
       "in the DRUG_HINT group ... reduction of 1.5%",
       "patients receiving DRUG_HINT had an HbA1c decrease of 1.3%",
       "DRUG_HINT reduced HbA1c by 1.2%".
  - Ignore changes clearly tied to other drugs or control groups.
  - Ignore pure between-group difference values like "mean difference -0.35% between DRUG_HINT and comparator".
  - If you cannot confidently tie any valid A1c reduction to DRUG_HINT, then:
       "extracted": []
       and do NOT set selected_percent (leave it empty or omit it).

- If DRUG_HINT is EMPTY:
  - You may extract A1c reductions for any clearly defined arm,
    but still must ignore responder rates, dose changes, and between-group differences.

NOTE ON RELATIVE REDUCTIONS:
- Relative reductions from "from X% to Y%" are handled by the surrounding application logic.
- You do NOT need to invent or compute new % values that are not explicitly present in the text.
- Focus on identifying which explicit % values correspond to A1c change magnitudes.

DURATION:
- From FULL_ABSTRACT_FOR_DURATION, extract the main trial/timepoint duration tied to the A1c result:
    e.g., "6 months", "26 weeks", "T12", "52 weeks", "24–52 weeks".
- If multiple timepoints are present, pick the one most clearly tied to the primary A1c outcome.

FINAL JSON:
- Must be valid JSON, parseable, and contain at least:
    "extracted": [...]   (can be an empty list)
- Do NOT include any text outside the JSON object.
"""

def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

LLM_DEBUG_SHOWN = False

def get_resp_text(resp):
    """Try multiple ways to extract text from a google.generativeai response."""
    txt = getattr(resp, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts
            chunks = []
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str):
                    chunks.append(t)
            if chunks:
                return "\n".join(chunks)
    except Exception:
        pass
    try:
        return str(resp)
    except Exception:
        return ""

def llm_extract_a1c_and_duration(
    model,
    a1c_sentences: str,
    full_abstract: str,
    drug_hint: str = "",
):
    """
    LLM reads:
      - shortlisted sentences for A1c values
      - full abstract for duration
      - optional drug hint

    IMPORTANT POST-FILTER:
    Any percent value suggested by the LLM that does NOT literally appear as a percent
    in `a1c_sentences` is discarded.
    Relative reductions (like 25%) that the LLM invents are dropped – relative reductions
    are handled by regex for 'from X% to Y%' patterns.
    """
    global LLM_DEBUG_SHOWN

    if (
        model is None
        or not a1c_sentences
        or not a1c_sentences.strip()
        or not full_abstract
        or not full_abstract.strip()
    ):
        return [], "", ""

    # Collect all numeric % actually present in the shortlisted sentences
    sentence_percent_nums = []
    for m in re_pct.finditer(a1c_sentences):
        n = parse_number(m.group(1))
        if not math.isnan(n):
            sentence_percent_nums.append(abs(n))

    def appears_in_sentences(num: float, tol: float = 0.01) -> bool:
        """Return True if num is (within tol) of any % value literally present in a1c_sentences."""
        for s in sentence_percent_nums:
            if abs(num - s) <= tol:
                return True
        return False

    header = ""
    if drug_hint:
        header += f"DRUG_HINT: {drug_hint}\n"
    else:
        header += "DRUG_HINT: \n"

    prompt = (
        header
        + "SENTENCES_FOR_A1C:\n"
        + a1c_sentences
        + "\n\nFULL_ABSTRACT_FOR_DURATION:\n"
        + full_abstract
        + "\n\n"
        + LLM_RULES
        + "\nReturn JSON only."
    )

    try:
        resp = model.generate_content(prompt)
        text = get_resp_text(resp).strip()

        if not LLM_DEBUG_SHOWN:
            with st.expander("LLM raw output (first call)"):
                st.write(text)
            LLM_DEBUG_SHOWN = True

        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e <= s:
            return [], "", ""

        data = json.loads(text[s : e + 1])

        # Filtered extracted values: keep only % appearing in shortlisted sentences
        extracted = []
        for x in (data.get("extracted") or []):
            if not isinstance(x, str):
                continue
            m = re.search(r'([+-]?\d+(?:[.,·]\d+)?)\s*%?', x)
            if not m:
                continue
            num = parse_number(m.group(1))
            if math.isnan(num):
                continue
            num_abs = abs(num)
            if appears_in_sentences(num_abs):
                extracted.append(fmt_pct(num_abs))

        # selected_percent (also filtered)
        selected = ""
        selected_raw = data.get("selected_percent", "") or ""
        if isinstance(selected_raw, str) and selected_raw.strip():
            m = re.search(r'([+-]?\d+(?:[.,·]\d+)?)\s*%?', selected_raw)
            if m:
                num = parse_number(m.group(1))
                if not math.isnan(num):
                    num_abs = abs(num)
                    if appears_in_sentences(num_abs):
                        selected = fmt_pct(num_abs)

        duration = ""
        if isinstance(data.get("duration"), str):
            duration = data.get("duration").strip()

        return extracted, selected, duration

    except Exception as e:
        st.error(f"Gemini LLM call failed: {e}")
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
def process_df(
    df_in: pd.DataFrame,
    text_col: str,
    model,
    use_llm: bool,
    drug_col_name: Optional[str],
):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, "")
        if not isinstance(text_orig, str):
            text_orig = "" if pd.isna(text_orig) else str(text_orig)

        # Drug hint per row
        drug_hint = ""
        if drug_col_name and drug_col_name in df_in.columns:
            drug_hint = str(row.get(drug_col_name, "") or "")

        duration_regex = extract_durations_regex(text_orig)

        hba_matches, hba_sentences = extract_sentences(text_orig)
        shortlisted = " | ".join(hba_sentences) if hba_sentences else ""

        # Precomputed relative from-to (from regex)
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
                        if not (math.isnan(a) or math.isnan(b)) and a != 0:
                            rel = ((a - b) / a) * 100.0
                            return fmt_pct(rel)
            return None

        precomputed_rel = _find_precomputed_relative(hba_matches)

        def fmt_extracted(m):
            t = (m.get("type") or "").lower()
            if "from-to" in t:
                if "relative" in t:
                    vals = (m.get("values") or [])
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

        # LLM extraction
        llm_extracted, llm_selected, llm_duration = [], "", ""
        if use_llm and model is not None and shortlisted:
            llm_extracted, llm_selected, llm_duration = llm_extract_a1c_and_duration(
                model, shortlisted, text_orig, drug_hint
            )

        final_duration = llm_duration if llm_duration else duration_regex

        # A1c reduction values string (LLM only)
        a1c_reduction_values_str = " | ".join(llm_extracted) if llm_extracted else ""

        # If no LLM A1c reduction values → selected % must be empty (no regex fallback)
        if not llm_extracted:
            selected = ""
        else:
            # Check if all LLM values are zero
            all_llm_zero = False
            nums = []
            for item in llm_extracted:
                if isinstance(item, str):
                    m = re.search(r'([+-]?\d+(?:[.,·]\d+)?)', item)
                    if m:
                        v = parse_number(m.group(1))
                        if not math.isnan(v):
                            nums.append(abs(v))
            if nums and all(v == 0 for v in nums):
                all_llm_zero = True

            if all_llm_zero:
                selected = fmt_pct(0.0)
            else:
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
                "sentence": shortlisted,
                "extracted_matches": hba_regex_vals,
                "LLM extracted": llm_extracted,
                "A1c reduction values": a1c_reduction_values_str,
                "selected %": selected,
                "A1c Score": a1c_score,
                "duration": final_duration,
            }
        )
        rows.append(new)

    out = pd.DataFrame(rows)

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
drug_col_name = st.sidebar.text_input(
    "Column with drug name (optional, for DRUG_HINT)", value="drug_name"
)
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

# if user typed a drug column that doesn't exist, ignore hint and warn
if drug_col_name and drug_col_name not in df.columns:
    st.warning(
        f'Drug column "{drug_col_name}" not found in data. '
        "Drug-based extraction will be skipped."
    )
    effective_drug_col = None
else:
    effective_drug_col = drug_col_name if drug_col_name in df.columns else None

model = None
if use_llm:
    if not GENAI_AVAILABLE:
        st.error("google.generativeai is not installed or not importable — LLM disabled.")
        use_llm = False
    else:
        key = API_KEY.strip()
        if not key:
            st.error("No GEMINI_API_KEY environment variable set. LLM disabled.")
            use_llm = False
        else:
            model = configure_gemini(key)
            if model is None:
                st.error("Failed to configure Gemini model (check key or network). LLM disabled.")
                use_llm = False

st.success(f"Loaded {len(df)} rows. Processing...")

out_df = process_df(df, col_name, model, use_llm, effective_drug_col)

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
