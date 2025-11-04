"""
extract_hba1c.py

Usage:
    python extract_hba1c.py input.xlsx --sheet "Sheet1" --col "abstract" --out output.xlsx

Outputs:
    - A new Excel file (or CSV) with one row per input row plus:
        * extracted_matches: list of raw matched strings
        * reductions_pp: list of numeric absolute reductions in percentage points (floats)
        * reduction_types: list describing how value was obtained (percent-match, from-to, pp, range)
        * best_reduction_pp: the single best reduction chosen (max absolute reduction) or NaN
"""

import re
import math
import argparse
from typing import List, Dict, Any, Tuple
import pandas as pd

# -- Regex building blocks
NUM = r'(?:\d+(?:[.,]\d+)?)'  # digits with optional decimal (comma or dot)
PCT = rf'({NUM})\s*%'  # capture percent like 1.2%
PP_WORD = r'(?:percentage points?|pp\b|points?)'
FROM_TO = rf'from\s+({NUM})\s*%\s*(?:to|->|−|—|-)\s*({NUM})\s*%'

# other phrasings:
REDUCE_BY = rf'(?:reduc(?:e|ed|tion|ed by|ing)|decrease(?:d)?|drop(?:ped)?|fell|reduced)\s*(?:by\s*)?({NUM})\s*(?:%|\s*{PP_WORD})'
ABS_PP = rf'(?:absolute\s+reduction\s+of|reduction\s+of)\s*({NUM})\s*(?:%|\s*{PP_WORD})'
RANGE_PCT = rf'({NUM})\s*(?:–|-|—)\s*({NUM})\s*%'

# compile with case-insensitive
FLAGS = re.IGNORECASE
re_pct = re.compile(PCT, FLAGS)
re_fromto = re.compile(FROM_TO, FLAGS)
re_reduce_by = re.compile(REDUCE_BY, FLAGS)
re_abs_pp = re.compile(ABS_PP, FLAGS)
re_range = re.compile(RANGE_PCT, FLAGS)

# helper to parse numbers with commas or dots
def parse_number(s: str) -> float:
    s = s.replace(',', '.').strip()
    try:
        return float(s)
    except:
        return float('nan')

def extract_hba1c_reductions(text: str) -> List[Dict[str,Any]]:
    """
    Returns a list of matches. Each match is a dict:
    {
      'raw': matched text,
      'type': 'from-to' | 'percent' | 'pp' | 'range',
      'values': list of floats (numbers parsed),
      'reduction_pp': numeric absolute reduction in percentage points (if computable), else None,
      'span': (start, end)
    }
    """
    if not isinstance(text, str):
        return []
    matches = []

    t = text

    # 1) from X% to Y%  -> compute X - Y as absolute reduction
    for m in re_fromto.finditer(t):
        a = parse_number(m.group(1))
        b = parse_number(m.group(2))
        reduction = None
        # If first is larger than second, reduction is positive (a -> b)
        if not math.isnan(a) and not math.isnan(b):
            reduction = a - b
        matches.append({
            'raw': m.group(0),
            'type': 'from-to',
            'values': [a, b],
            'reduction_pp': reduction,
            'span': m.span()
        })

    # 2) explicit "reduced by X%" or "decrease X%" or "reduction of X percentage points"
    for m in re_reduce_by.finditer(t):
        v = parse_number(m.group(1))
        # ambiguous whether it's percent or percentage points: treat as absolute pp
        matches.append({
            'raw': m.group(0),
            'type': 'percent_or_pp',
            'values': [v],
            'reduction_pp': v,
            'span': m.span()
        })

    # 3) absolute pp phrasing
    for m in re_abs_pp.finditer(t):
        v = parse_number(m.group(1))
        matches.append({
            'raw': m.group(0),
            'type': 'pp_word',
            'values': [v],
            'reduction_pp': v,
            'span': m.span()
        })

    # 4) ranges like "1.0–1.5%"
    for m in re_range.finditer(t):
        a = parse_number(m.group(1))
        b = parse_number(m.group(2))
        # choose the larger value as the likely 'max reduction' in the range; store both
        matches.append({
            'raw': m.group(0),
            'type': 'range_percent',
            'values': [a, b],
            'reduction_pp': max(a, b) if not math.isnan(a) and not math.isnan(b) else None,
            'span': m.span()
        })

    # 5) plain percent mentions in proximity to HbA1c words:
    # Restrict to percent matches near "HbA1c" or "A1c" within the same sentence or short window.
    # We'll find percent tokens and then check context.
    pct_hits = []
    for m in re_pct.finditer(t):
        pct_hits.append((m, parse_number(m.group(1))))

    # find occurrences of "HbA1c" / "A1c" to restrict context
    hba_words = list(re.finditer(r'\b(?:hba1c|hb a1c|a1c)\b', t, FLAGS))
    if hba_words:
        # for each percent, check if there's an hba word within +/- 80 characters
        for m, val in pct_hits:
            start, end = m.span()
            near = any( abs(start - hh.span()[0]) <= 80 or abs(end - hh.span()[1]) <= 80 for hh in hba_words )
            if near:
                matches.append({
                    'raw': m.group(0),
                    'type': 'percent_near_hba1c',
                    'values': [val],
                    'reduction_pp': val,
                    'span': m.span()
                })
    else:
        # no explicit HbA1c word; still collect percent matches but mark them 'global_percent'
        for m, val in pct_hits:
            matches.append({
                'raw': m.group(0),
                'type': 'percent_global',
                'values': [val],
                'reduction_pp': val,
                'span': m.span()
            })

    # Post-process: deduplicate overlapping matches by span (prefer more specific types)
    # keep order of detection but filter exact duplicate spans
    seen_spans = set()
    filtered = []
    for mm in matches:
        sp = mm['span']
        if sp in seen_spans:
            continue
        seen_spans.add(sp)
        filtered.append(mm)

    # sort by span start
    filtered.sort(key=lambda x: x['span'][0])
    return filtered

def choose_best_reduction(matches: List[Dict[str,Any]]) -> Tuple[float, str]:
    """
    Choose a single 'best' reduction from a list of matches.
    Strategy: pick the largest numeric absolute reduction_pp (if present).
    Returns (value, reason)
    """
    best = None
    reason = ''
    for m in matches:
        v = m.get('reduction_pp')
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if best is None or abs(v) > abs(best):
            best = v
            reason = m.get('type', '')
    return (best, reason)

def process_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        text = row.get(text_col, '')
        matches = extract_hba1c_reductions(text)
        extracted_raws = [m['raw'] for m in matches]
        reductions = [m.get('reduction_pp') for m in matches]
        types = [m.get('type') for m in matches]
        best, reason = choose_best_reduction(matches)
        new = row.to_dict()
        new.update({
            'extracted_matches': extracted_raws,
            'reductions_pp': reductions,
            'reduction_types': types,
            'best_reduction_pp': best,
            'best_reduction_reason': reason
        })
        rows.append(new)
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input Excel file (xlsx or csv)')
    parser.add_argument('--sheet', default=None, help='sheet name for xlsx (optional)')
    parser.add_argument('--col', default='abstract', help='column name that contains abstracts/text')
    parser.add_argument('--out', default='extracted_hba1c.xlsx', help='output file (xlsx or csv)')
    args = parser.parse_args()

    if args.input.lower().endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input, sheet_name=args.sheet)

    if args.col not in df.columns:
        print(f"ERROR: column '{args.col}' not found. Available columns: {list(df.columns)}")
        return

    out_df = process_dataframe(df, args.col)

    if args.out.lower().endswith('.csv'):
        out_df.to_csv(args.out, index=False)
    else:
        out_df.to_excel(args.out, index=False)
    print("Done. Saved to", args.out)

if __name__ == '__main__':
    main()
