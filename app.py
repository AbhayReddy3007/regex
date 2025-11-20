# streamlit_hba1c_weight_llm.py
# Updated: stricter LLM rules + post-validation so the LLM cannot return irrelevant percentages.
# - LLM instructed to extract only target-related change percentages.
# - Post-validation uses the regex extractor to allow only percents that are actually present
#   in the sentence near the target (or the computed relative reduction).

import re
import math
import json
from io import BytesIO

import pandas as pd
import streamlit as st

# ===================== HARD-CODE YOUR GEMINI KEY HERE =====================
API_KEY = ""   # <- replace with your real key if you want LLM enabled
# =========================================================================

# Lazy Gemini import so the app still runs without the package
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
    space_like = set([' ', '\t', '\n', '\r'])
    L = len(s)

    # Left side
    i = pos - 1
    spaces = 0
    left_boundary_start = pos  # default
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

    Note: For tag_prefix == 'hba1c' we compute relative reduction ((X-Y)/X)*100 for from->to.
    """
    matches = []

    # 1) WHOLE-SENTENCE: "from X% to Y%"
    for m in re_fromto.finditer(sent):
        a = parse_number(m.group(1))  # original X
        b = parse_number(m.group(2))  # new Y
        reduction_val = None
        if (not math.isnan(a)) and (not math.isnan(b)) and (a != 0):
            try:
                if tag_prefix == 'hba1c':
                    # relative reduction %
                    reduction_val = ((a - b) / a) * 100.0
                else:
                    # keep absolute percentage-point difference for non-HbA1c (legacy behavior)
                    reduction_val = (a - b)
            except Exception:
                reduction_val = None
        add_match(matches, si, 0, m, f'{tag_prefix}:from-to_sentence', [a, b], reduction_val)

    # 2) ±5-SPACES window (inclusive) around each target term: other patterns only
    any_window_hit = False
    for hh in term_re.finditer(sent):
        seg, abs_s, _ = window_prev_next_spaces_inclusive_tokens(sent, hh.end(), 5, 5)

        # reduced/decreased/... by X%
        for m in re_reduce_by.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:percent_or_pp_pmSpaces5', [v], v)

        # reduction of X%
        for m in re_abs_pp.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:pp_word_pmSpaces5', [v], v)

        # ranges like 1.0–1.5% (represent as max)
        for m in re_range.finditer(seg):
            any_window_hit = True
            a = parse_number(m.group(1)); b = parse_number(m.group(2))
            rep = None if (math.isnan(a) or math.isnan(b)) else max(a, b)
            add_match(matches, si, abs_s, m, f'{tag_prefix}:range_percent_pmSpaces5', [a, b], rep)

        # any stray percent in the window
        for m in re_pct.finditer(seg):
            any_window_hit = True
            v = parse_number(m.group(1))
            add_match(matches, si, abs_s, m, f'{tag_prefix}:percent_pmSpaces5', [v], v)

    # 3) Weight safety: nearest previous % within 60 chars if no window hit
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
    """Return (matches, sentences_used) for sentences meeting the criterion.

    NOTE: we also include sentences that contain a 'from X% to Y%' pattern if the target
    term appears in the previous or next sentence (to capture 'from X to Y' phrasing that
    is split across sentences).
    """
    matches, sentences_used = [], []
    sents = split_sentences(text)
    for si, sent in enumerate(sents):
        keep = False
        if sentence_meets_criterion(sent, term_re):
            keep = True
        else:
            if re_fromto.search(sent):
                if term_re.search(sent):
                    keep = True
                else:
                    prev_has = (si > 0) and bool(term_re.search(sents[si-1]))
                    next_has = (si+1 < len(sents)) and bool(term_re.search(sents[si+1]))
                    if prev_has or next_has:
                        keep = True
        if not keep:
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

# -------------------- UI --------------------
st.sidebar.header('Options')
uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
drug_col   = st.sidebar.text_input('Column with drug name (optional)', value='drug_name')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')
show_debug = st.sidebar.checkbox('Show debug columns (reductions_pp, reduction_types)', value=False)
use_llm   = st.sidebar.checkbox('Enable Gemini LLM (Gemini 2.0 Flash)', value=True)

if not uploaded:
    st.info('Upload your Excel or CSV file in the left sidebar. Example: my_abstracts.xlsx with column named "abstract".')
    st.stop()

# read file robustly and handle common encodings
try:
    if uploaded.name.lower().endswith('.csv'):
        try:
            df = pd.read_csv(uploaded, encoding='utf-8')
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding='utf-8', error_bad_lines=False, engine='python')
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

# -------------------- Gemini 2.0 Flash setup --------------------
def configure_gemini(api_key: str):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

model = configure_gemini(API_KEY) if use_llm else None

def _norm_percent(v: str) -> str:
    v = (v or "").strip().replace(" ", "")
    if v and not v.endswith("%"):
        if re.match(r"^[+-]?\d+(?:[.,·]\d+)?$", v):
            v += "%"
    return v

# -------------------- STRICTER LLM RULES --------------------
LLM_RULES = """
You are an information-extraction assistant.  Read the SENTENCE(S) provided and extract ONLY percentage *changes* that refer to the specified TARGET.

REPLY RULE (MANDATORY)
- Return STRICT JSON ONLY, exactly with these keys and no others:
  {
    "extracted": ["...%", "...%"],
    "selected_percent": "...%"
  }
- Do not add any extra text outside the JSON object.
- If no valid percentage-change applies, return:
  {
    "extracted": [],
    "selected_percent": ""
  }

DEFINITIONS
- TARGET will be one of: "HbA1c", "A1c", or "Body weight".
- A *percentage-change* is any percent value that indicates a change (reduction/decline/decrease/drop/etc.), a percentage-point change expressed as "reduction of X%", or an indicated range of change like "1.2–1.5%".
- Do NOT return baseline-only percentages (e.g., "baseline 8.2%") unless they are explicitly presented as part of a change (e.g., "from 8.2% to 7.1%").

NORMALIZATION
- All values in "extracted" and "selected_percent" must be strings that:
  - include a trailing percent sign `%`
  - use a dot `.` decimal separator (convert commas `,` or middle dots `·` to `.`)
  - are trimmed of whitespace
  - contain only digits, optional leading sign, optional decimal part, then `%` (no “<”, “>”, “≥”, “≤”, or words)
- Preferred formatting: up to 3 decimal places is fine (e.g., `11.538%`); trailing zeros may be trimmed (`11.5%` allowed). Always include `%`.

PRIMARY EXTRACTION RULES
1. Extract only percentages that clearly refer to the TARGET and describe a CHANGE (reduced/decreased/declined/dropped/lowered/fell/was reduced by / from X to Y / absolute reduction of X% / X–Y% range).
2. If the sentence contains multiple percentages, include them all in "extracted" **only if** they are clearly change values for the TARGET (see proximity rule below).
3. For ranges (e.g., "1.2–1.5%", "1.2 to 1.5%", "1.2—1.5%"):
   - Include **both** endpoints in "extracted" as separate percent strings (e.g., `["1.2%", "1.5%"]`).
   - When choosing `selected_percent`, prefer the **higher endpoint** (e.g., choose `1.5%`) unless a computed relative reduction (see below) applies which overrides this preference.

SPECIAL: `from X% to Y%` FOR HbA1c/A1c (MANDATORY)
- If you see a `from X% to Y%` pattern that clearly refers to the TARGET **and** the TARGET is HbA1c/A1c:
  - You MUST compute the **relative reduction** as:
      ((X - Y) / X) * 100
  - Format that computed value as a percent string and:
      - include it in "extracted"
      - set it as "selected_percent"
  - **Do not** set `selected_percent` to `(X - Y)` (the absolute difference). Return the computed relative percent.
  - Example:
      Sentence: "HbA1c improved from 8.2% to 7.1%."
      X=8.2, Y=7.1 -> relative=((8.2-7.1)/8.2)*100 -> `"13.415%"`
      Output: {"extracted":["13.415%"], "selected_percent":"13.415%"}

PROXIMITY / CONTEXT RULE (AVOID HALLUCINATION)
- Only extract a percent if it is:
  - in the same sentence as the TARGET and a reduction cue, OR
  - in a very nearby clause that explicitly links it to the TARGET (e.g., "for HbA1c, from X% to Y%"), OR
  - part of the same short paragraph where the text unambiguously ties it to the TARGET.
- Do NOT extract general/statistical percentages unrelated to the TARGET (population %, p-values not representing change, confidence intervals, eligibility thresholds like "<7%").
- If a percentage is ambiguous in context, do NOT extract it.

DOSE / MULTIPLE-DOSE RULE
- If multiple doses are reported (e.g., "0.5 mg yielded 1.2–1.5% while 1.0 mg yielded 1.4–1.8%"):
  - Extract all change values (as per rules) in "extracted".
  - Prefer the change for the **highest dose** when choosing "selected_percent" (so pick `1.8%` for the 1.0 mg example).
  - If highest-dose association cannot be determined confidently, fall back to the highest numeric percent.

RANGE HANDLING (explicit)
- For text like "1.2–1.5%": add `"1.2%"` and `"1.5%"` to "extracted".
- When deciding `selected_percent` among a range, pick the **higher endpoint**.
- If there are multiple ranges, apply dose preference first; otherwise choose the overall numeric maximum.

TIE-BREAKERS (ORDER OF PREFERENCE)
When multiple candidate percent-changes could be selected, choose in this order:
1. Computed relative reduction from a `from X% to Y%` (HbA1c override).
2. Change associated with the highest dose (if doses are explicitly mentioned).
3. Highest numeric percent among valid candidates.
4. If still ambiguous, choose the value most directly adjacent to the TARGET term (closest by character distance).
5. If ambiguity remains, return the highest numeric percent.

ERROR / NOISE HANDLING
- If the model cannot confidently find any qualifying percent-change for the TARGET, return:
  {"extracted": [], "selected_percent": ""}
- Do not invent or guess percentages not present in the text.
- Do not return p-values, CI bounds, baseline-only percentages, or thresholds (e.g., "<7%") as change values.

VALID EXAMPLES (required outputs — follow EXACTLY):
1) Input sentence: "HbA1c levels declined from 7.80% to 6.90%."
   Output: {"extracted":["11.538%"], "selected_percent":"11.538%"}
   (computed relative reduction = ((7.8-6.9)/7.8)*100 = 11.538%)

2) Input: "From a baseline range of 8.1-8.7 %, 0.5 mg dose lowered HbA1c by 1.2-1.5 %, while the 1.0 mg dose reduced it by 1.4-1.8 %."
   Extracted: include ["1.2%", "1.5%", "1.4%", "1.8%"]
   selected_percent: "1.8%" (highest endpoint for highest dose 1.0 mg)

3) Input: "Body weight decreased by 12.3% (p<0.001)."
   Output: {"extracted":["12.3%"], "selected_percent":"12.3%"}

4) Input: "HbA1c decreased by 1.1% from 8.2% at baseline."
   - This is **ambiguous** if phrase doesn't indicate from->to; treat as percentage-change: extracted ["1.1%"], selected "1.1%".
   - But if text says "HbA1c improved from 8.2% to 7.1%", then use computed relative reduction (example 1).

FORMAT RULES – numeric normalization
- Convert `,` or `·` decimal separators to `.`
- Accept leading `+` or `-` but return absolute value (positive percent) in output.
- Remove ranges punctuation and output endpoints separately.
- Drop any percent that contains `<`, `>`, `≤`, `≥` symbols (these are thresholds, not change values).

OUTPUT STRICTNESS
- The JSON must be parseable by a strict JSON parser.
- Do not include comments, trailing commas, or any non-JSON text.
- Example exact output forms shown above must be followed.

FINAL NOTE
- Your system will validate the returned JSON and may forcibly replace the LLM result with a computed relative reduction for `from X% to Y%` if the model fails to follow instructions. Still, do your best: **compute the relative reduction yourself when asked** and make that the `selected_percent`.

END OF RULES
"""


def _allowed_percents_from_regex(sentence: str, target_label: str):
    """
    Run the extraction regex on the sentence and return a set of normalized percent strings
    that are considered 'allowed' (i.e., relevant to the target).
    """
    allowed = set()
    # choose term_re and tag
    if target_label.lower().startswith('hba1c') or target_label.lower().startswith('a1c'):
        term_re, tag = re_hba1c, 'hba1c'
    else:
        term_re, tag = re_weight, 'weight'

    # get matches for this sentence using existing logic
    matches, sents = extract_sentences(sentence, term_re, tag)
    for m in matches:
        # if from-to type, use computed reduction_pp (already computed in extract_in_sentence)
        t = (m.get('type') or '').lower()
        if 'from-to' in t:
            rp = m.get('reduction_pp')
            if rp is not None and not (isinstance(rp, float) and math.isnan(rp)):
                allowed.add(fmt_pct(rp))
            # also include raw endpoints
            vals = m.get('values') or []
            for v in vals:
                try:
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        allowed.add(fmt_pct(v))
                except:
                    pass
        elif 'range' in t:
            vals = m.get('values') or []
            if len(vals) >= 2:
                a, b = vals[0], vals[1]
                if not (math.isnan(a) or math.isnan(b)):
                    allowed.add(fmt_pct(a))
                    allowed.add(fmt_pct(b))
                    # include max endpoint explicitly
                    allowed.add(fmt_pct(max(a, b)))
        else:
            vals = m.get('values') or []
            for v in vals:
                if v is None:
                    continue
                if isinstance(v, (int, float)):
                    if not math.isnan(v):
                        allowed.add(fmt_pct(v))
                else:
                    try:
                        num = parse_number(v)
                        if not math.isnan(num):
                            allowed.add(fmt_pct(num))
                    except:
                        pass

    return allowed

def llm_extract_from_sentence(model, target_label: str, sentence: str, drug_hint: str = ""):
    """
    Conservative LLM extraction + post-validation.
    - Instructs the LLM (via LLM_RULES) to compute relative reductions when applicable.
    - Parses LLM JSON output.
    - Validates: only keep LLM-extracted percents that are present in regex-based allowed set OR are the computed relative reduction.
    - If computed relative reduction exists, prefer it (force if necessary).
    """
    if model is None or not sentence.strip():
        return [], ""

    # precompute authoritative computed reductions (for HbA1c)
    computed_reductions = []
    if (target_label.lower().startswith('hba1c') or target_label.lower().startswith('a1c')):
        for m_ft in re_fromto.finditer(sentence):
            a = parse_number(m_ft.group(1)); b = parse_number(m_ft.group(2))
            if not (math.isnan(a) or math.isnan(b)) and a != 0:
                try:
                    rel = ((a - b) / a) * 100.0
                    computed_reductions.append(fmt_pct(rel))
                except:
                    pass

    # call the LLM (if available)
    prompt = (
        f"TARGET: {target_label}\n"
        + (f"DRUG_HINT: {drug_hint}\n" if drug_hint else "")
        + LLM_RULES + "\n"
        + "SENTENCE:\n" + sentence + "\n"
        + "Return JSON only.\n"
    )

    text = ""
    try:
        resp = model.generate_content(prompt) if model is not None else None
        text = (getattr(resp, "text", "") or "").strip() if resp is not None else ""
    except Exception:
        text = ""

    # parse LLM JSON safely
    extracted_raw = []
    selected_raw = ""
    try:
        if text:
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e > s:
                data = json.loads(text[s:e+1])
                raw_ex = data.get("extracted") or []
                for x in raw_ex:
                    if isinstance(x, str):
                        extracted_raw.append(x.strip())
                selected_raw = (data.get("selected_percent") or "") or ""
    except Exception:
        extracted_raw = []
        selected_raw = ""

    # normalize LLM outputs
    extracted = []
    for x in extracted_raw:
        xn = _norm_percent(x)
        if re.search(r'[<>≥≤]', xn):
            continue
        extracted.append(xn)
    selected = _norm_percent(selected_raw)

    # Build allowed set from regex extraction of the sentence
    allowed_set = _allowed_percents_from_regex(sentence, target_label)

    # If computed_reductions exist, prefer/force them (computed_reductions already formatted)
    if computed_reductions:
        comp = computed_reductions[0]
        # If LLM already returned comp correctly, keep it; otherwise force it
        if comp not in extracted or selected != comp:
            # ensure comp is at front of extracted
            if comp in extracted:
                extracted.remove(comp)
            extracted.insert(0, comp)
            selected = comp
        # final filter: keep only allowed items + comp (comp is already allowed by policy)
        # ensure comp is present in allowed (we treat computed as allowed)
        # return immediately — computed reduction trumps noisy LLM extractions
        return [e for e in extracted if re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', e)], selected

    # No computed reduction: filter LLM outputs to only values present in allowed_set
    if allowed_set:
        extracted_filtered = [e for e in extracted if e in allowed_set]
        # If selected is not in allowed, drop it
        if selected not in allowed_set:
            selected = ""
        extracted = extracted_filtered

    # If nothing remains after strict filtering, fallback to proximity-based conservative filter:
    # keep LLM extracted percents that occur within 120 chars of a target keyword occurrence in sentence.
    if not extracted:
        # find positions of target terms
        term_re = re_hba1c if target_label.lower().startswith('hba1c') or target_label.lower().startswith('a1c') else re_weight
        positions = [m.start() for m in term_re.finditer(sentence)]
        if positions:
            prox_allowed = []
            for ex in extracted:
                # find the numeric version and search its occurrence in the sentence
                ex_plain = ex.replace('%', '')
                # search for a matching percent near any target position
                found_near = False
                for p in positions:
                    left = max(0, p - 120)
                    right = min(len(sentence), p + 120)
                    window = sentence[left:right]
                    if re.search(re.escape(ex_plain) + r'\s*%', window):
                        found_near = True
                        break
                if found_near:
                    prox_allowed.append(ex)
            extracted = prox_allowed
            if selected and selected not in extracted:
                selected = ""

    # If still empty and extracted is empty, as a last resort keep nothing (conservative)
    # but if LLM returned something and user prefers to see any candidate, that would need toggling.
    # For now be strict: if extracted empty -> return empty.
    if not extracted:
        return [], ""

    # If selected empty, pick numeric max across extracted (consistent with earlier behavior)
    if not selected:
        max_val = None
        for ex in extracted:
            try:
                num = parse_number(ex.replace('%', ''))
                if math.isnan(num):
                    continue
                if max_val is None or num > max_val:
                    max_val = num
            except:
                continue
        if max_val is not None:
            selected = fmt_pct(max_val)

    # Final cleanup: ensure percent formatting
    extracted = [e for e in extracted if re.match(r'^[+-]?\d+(?:[.,·]\d+)?%$', e)]
    return extracted, selected

# -------------------- Scoring helpers --------------------
def compute_a1c_score(selected_pct_str: str):
    """Scores for A1c:
    5: >2.2%
    4: 1.8%-2.1%
    3: 1.2%-1.7%
    2: 0.8%-1.1%
    1: <0.8%
    """
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
    """Scores for weight:
    5: >=22%
    4: 16-21.9%
    3: 10-15.9%
    2: 5-9.9%
    1: <5%
    """
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

# -------------------- Processing function (note _model to avoid caching errors) --------------------
@st.cache_data
def process_df(_model, df_in: pd.DataFrame, text_col: str, drug_col_name: str):
    rows = []
    for _, row in df_in.iterrows():
        text_orig = row.get(text_col, '')
        if not isinstance(text_orig, str):
            text_orig = '' if pd.isna(text_orig) else str(text_orig)

        # Run regex extraction to produce the 'sentence' columns (only sentences that meet the strict criteria)
        hba_matches, hba_sentences = extract_sentences(text_orig, re_hba1c, 'hba1c')
        wt_matches, wt_sentences   = extract_sentences(text_orig, re_weight, 'weight')

        # Format regex outputs
        def fmt_extracted(m):
            t = (m.get('type') or '').lower()
            if 'from-to' in t:
                # present as percent (for HbA1c it's relative reduction, for weight it's kept as stored)
                return fmt_pct(m.get('reduction_pp'))
            return m.get('raw', '')

        hba_regex_vals = [fmt_extracted(m) for m in hba_matches]
        wt_regex_vals  = [fmt_extracted(m) for m in wt_matches]

        # Build sentence strings (join qualifying sentences with ' | ')
        sentence_str = ' | '.join(hba_sentences) if hba_sentences else ''
        weight_sentence_str = ' | '.join(wt_sentences) if wt_sentences else ''

        # LLM extraction: read from the sentence column and produce LLM extracted + selected %
        drug_hint = ""
        if drug_col_name and drug_col_name in df_in.columns:
            drug_hint = str(row.get(drug_col_name, '') or "")

        # HbA1c LLM
        hba_llm_extracted, hba_selected = ([], "")
        if _model is not None and sentence_str:
            hba_llm_extracted, hba_selected = llm_extract_from_sentence(_model, "HbA1c", sentence_str, drug_hint)

        # Weight LLM
        wt_llm_extracted, wt_selected = ([], "")
        if _model is not None and weight_sentence_str:
            wt_llm_extracted, wt_selected = llm_extract_from_sentence(_model, "Body weight", weight_sentence_str, drug_hint)

        # If LLM extracted is empty, we must ensure selected remains empty (explicit requirement)
        if not hba_llm_extracted:
            hba_selected = ""
        if not wt_llm_extracted:
            wt_selected = ""

        # Ensure selected% is formatted and positive
        def normalize_selected(s):
            if not s:
                return ""
            s2 = s.replace('%', '').replace(',', '.').strip()
            try:
                v = float(s2)
                v = abs(v)  # make positive on user's request
                return fmt_pct(v)
            except:
                return ""

        hba_selected = normalize_selected(hba_selected)
        wt_selected  = normalize_selected(wt_selected)

        # Scores
        a1c_score = compute_a1c_score(hba_selected)
        weight_score = compute_weight_score(wt_selected)

        new = row.to_dict()
        new.update({
            'sentence': sentence_str,
            'extracted_matches': hba_regex_vals,
            'LLM extracted': hba_llm_extracted,
            'selected %': hba_selected,
            'A1c Score': a1c_score,
            'weight_sentence': weight_sentence_str,
            'weight_extracted_matches': wt_regex_vals,
            'Weight LLM extracted': wt_llm_extracted,
            'Weight selected %': wt_selected,
            'Weight Score': weight_score,
        })
        rows.append(new)

    out = pd.DataFrame(rows)

    # keep row if hba or weight has extracted matches (regex), or if LLM extracted exist (we assume we want rows with any data)
    def has_items(x):
        return isinstance(x, list) and len(x) > 0

    mask_hba = (out['sentence'].astype(str).str.len() > 0) & (out['extracted_matches'].apply(has_items))
    mask_wt  = (out['weight_sentence'].astype(str).str.len() > 0) & (out['weight_extracted_matches'].apply(has_items))
    mask_keep = mask_hba | mask_wt
    out = out[mask_keep].reset_index(drop=True)

    # counts for UI
    out.attrs['counts'] = dict(
        kept=int(mask_keep.sum()),
        total=int(len(mask_keep)),
        hba_only=int((mask_hba & ~mask_wt).sum()),
        wt_only=int((mask_wt & ~mask_hba).sum()),
        both=int((mask_hba & mask_wt).sum()),
    )

    return out

# Run processing (note we pass `model` in but process_df signature uses _model — avoids caching hash error)
out_df = process_df(model, df, col_name, drug_col)

# -------------------- Reorder columns: place LLM columns BESIDE regex columns --------------------
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
# keep user columns order + these additions
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
def to_excel_bytes(df_out):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

excel_bytes = to_excel_bytes(display_df)
st.download_button(
    'Download results as Excel',
    data=excel_bytes,
    file_name='results_with_llm_from_sentence.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
