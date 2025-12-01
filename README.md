The proportion of subjects with any decrease in HbA1c was higher among the once-weekly compared with the daily dose (82.4% vs.

Patients experienced mean decreases in HbA1C and weight from baseline to 6 months of -1.75% (P< 0.001) and -3.64 kg (P= 0.015), respectively, in the oral semaglutide group and -1.35% (P< 0.001) and -5.26 kg (P< 0.001), respectively, in the injectable semaglutide group. | When directly comparing semaglutide formulations, oral semaglutide demonstrated a 0.4% greater numerical reduction in HbA1C(P= 0.523) and injectable semaglutide demonstrated a 1.62-kg greater numerical reduction in weight (P= 0.312).

Patients experienced mean decreases in HbA1C and weight from baseline to 6 months of -1.75% (P< 0.001) and -3.64 kg (P= 0.015), respectively, in the oral semaglutide group and -1.35% (P< 0.001) and -5.26 kg (P< 0.001), respectively, in the injectable semaglutide group.

When adjusted for age and BMI, body weight and HbA1c reduction were not significantly different between the two formulations, as the proportion of patients achieving the composite outcome of weight loss ≥ 5% and HbA1c < 7.0%.

This study assessed the real-world effectiveness of available GLP-1 RAs in Romania on glycemic control, body weight reduction (BWR), and waist circumference (WC) in T2DM patients with excess weight.Methods: A prospective observational study was conducted on 311 adults with T2DM (glycated hemoglobin (HbA1c) > 7.2%, body mass index (BMI) ≥ 25 kg/m2). | Dulaglutide had the most significant impact on HbA1c (-6.69 ± 0.91%), while injectable semaglutide led to the most notable BWR (-4.60 ± 2.74 kg) and WC reduction, especially among male patients.

From a baseline range of 8.1-8.7 %, 0.5 mg dose lowered HbA1c by 1.2-1.5 %, while the 1.0 mg dose reduced it by 1.4-1.8 %.

HbA1c improved from 8.2 % to 7.1 %, with total daily insulin dose decreasing from 1.4 to 0.7 IU/kg/day (p < 0.001).

Median weight reduced from 100.0 kg to 91.5 kg (p<0.001), and median BMI decreased from 33.6 to 30.9 kg/m² (p<0.001). | HbA1c levels declined from 7.80% to 6.90% (p<0.001).

LLM_RULES = """
You are a strict, conservative information-extraction assistant. Read the provided SENTENCE(S) and EXTRACT ONLY percentage CHANGES that refer to the TARGET (one of: "HbA1c", "A1c", "Body weight").

**OUTPUT (MANDATORY, JSON ONLY)**
Return EXACTLY a single JSON object and nothing else, with these keys:
{
  "extracted": ["...%", "...%"],    // array of percent strings (each must exist in the sentence OR be computed from a present from->to)
  "selected_percent": "...%"        // single percent string chosen as the best candidate (or "" if none)
}

**ABSOLUTE RULES (do not break)**
1. ONLY include percentages that are:
   a) explicitly present in the SENTENCE as percent tokens (e.g., "12.3%") and clearly tied to the TARGET, OR
   b) explicitly computable from a present "from X% to Y%" or "from X kg to Y kg" in the sentence using the formulas below.
   If a percent is not present in the sentence and not computable from a present from->to, DO NOT include it.

2. When you see "from X% to Y%" and the TARGET is HbA1c/A1c:
   - Compute relative reduction = ((X - Y) / X) * 100
   - Include the computed percent in "extracted" and set it as "selected_percent".
   - Do NOT include (X - Y) as the selected percent.

3. When you see "from x kg to y kg" and the TARGET is Body weight:
   - Compute relative reduction using the agreed formula: ((x - y) / x) * 100
   - Include the computed percent in "extracted" and set it as "selected_percent".
   - Only compute if both x and y numeric kg values appear in the SENTENCE.

4. For ranges like "1.2–1.5%" or "1.2 to 1.5%":
   - Add both endpoints to "extracted": e.g. ["1.2%", "1.5%"].
   - When selecting a single value for "selected_percent" prefer the **higher endpoint** (e.g. "1.5%") unless a computed from->to relative reduction exists which overrides this.

5. Proximity / contextual requirement:
   - A percent token can be extracted **only** if the SENTENCE:
     * contains the TARGET term (or an immediately adjacent sentence ties the percent explicitly to the TARGET), and
     * contains a reduction cue (words like "reduced", "decreased", "fell", "lowered", "from", "declined", "dropped" etc.), OR the percent is part of a from->to pattern.
   - If the percent appears in the same sentence but is clearly unrelated (e.g., "50% of patients", "p=0.03", "CI 95%"), DO NOT extract it.

6. Dose handling:
   - If multiple doses are reported (e.g. "0.5 mg ... 1.0 mg ..."), and you have percentages tied to different doses:
     * Extract all valid percent-changes.
     * Prefer the percent associated with the **highest dose** when choosing "selected_percent". If dose association cannot be clearly established, choose the highest numeric percent.

7. Verification rule (critical):
   - For every percent you plan to add to "extracted":
     * Either locate the exact percent token in the sentence (match must be substring-equal after normalization), OR
     * It must be the result of a computation from an explicit from->to present in the same sentence.
   - If you cannot demonstrate either, DO NOT extract the percent.

8. Formatting / normalization:
   - All returned percent strings must:
     * Use `.` as decimal separator.
     * Contain only digits, optional leading sign, optional decimal part, then `%`. No `<`, `>`, `≤`, `≥`, `~`, or extra text.
     * Up to 3 decimal places is acceptable. Trailing zeros may be trimmed.
     * Always be positive (return absolute value).
   - Example allowed: "11.538%", "1.5%", "2%"

9. Ambiguity and refusal:
   - If you are not >= 90% confident a percent conforms to the rules above (i.e., clearly appears in context or is clearly computable), return:
     {"extracted": [], "selected_percent": ""}

10. Do not ever return:
   - p-values (e.g., "p < 0.001"), CI bounds, thresholds like "<7%", "% of patients", or any percent that is not specifically a change magnitude tied to the TARGET.

**Parsing details the model must perform (explicit instructions)**
- Normalize decimals: convert `,` or `·` to `.` before numeric parsing.
- When locating percent tokens in the sentence check for exact match: number (optionally decimal) followed immediately or shortly by optional space and `%`.
- When computing from->to:
  * For percent from->to: parse X and Y as numbers; require X != 0 for division; compute ((X-Y)/X)*100.
  * For kg from->to (Body weight): parse x and y as numbers in kg; require x != 0; compute ((x - y)/x)*100.
- For ranges written with hyphens/dashes or "to", split into endpoints and include both endpoints in extracted.

**Selection / tie-breaking (explicit priority when choosing selected_percent)**
1. Computed relative reduction from a present from->to (HbA1c percent from->to or Body weight kg from->to).
2. Change associated with highest dose (if doses are explicit).
3. Highest numeric percent among valid extracted candidates.
4. Closest percent to the TARGET term (character distance) as final tiebreaker.
5. If still ambiguous, return empty selected_percent.

**Required examples (model must follow these EXACT outputs)**

1) Sentence: "HbA1c improved from 8.2% to 7.1%."
   Output: {"extracted":["13.415%"], "selected_percent":"13.415%"}
   (because ((8.2-7.1)/8.2)*100 = 13.4146 -> "13.415%")

2) Sentence: "From baseline 100.0 kg to 91.5 kg, median weight decreased."
   Output: {"extracted":["8.5%"], "selected_percent":"8.5%"}
   (compute ((100 - 91.5)/100)*100 = 8.5%)

3) Sentence: "0.5 mg dose lowered HbA1c by 1.2-1.5%, while 1.0 mg reduced it by 1.4-1.8%."
   Output: {"extracted":["1.2%","1.5%","1.4%","1.8%"], "selected_percent":"1.8%"}
   (all endpoints extracted, highest endpoint for highest dose selected)

4) Sentence: "Body weight decreased by 12.3% (p<0.001)."
   Output: {"extracted":["12.3%"], "selected_percent":"12.3%"}

**Negative examples (must produce empty result)**
- "p=0.03, weight reduction significant." => {"extracted": [], "selected_percent": ""}
- "50% of patients achieved response." => {"extracted": [], "selected_percent": ""}
- "HbA1c baseline was 8.2%." (no change indicated) => {"extracted": [], "selected_percent": ""}

**Final instructions**
- Return strictly formatted JSON only and nothing else.
- You will be validated by downstream code: values not present in the sentence or not computed from explicit from->to will be rejected — so follow the verification steps above.
- If in doubt, prefer returning an empty extraction rather than a possibly incorrect percent.

END OF RULES
"""

Initially, insulin therapy was started, and her HbA1c decreased to 6.7% over 6 months. | By 27 months, her HbA1c improved to 5.1%, and overall, she had a 12% reduction in her percent above the 95th percentile (%BMIp95; BMI 41 kg/m²→35 kg/m²; %BMIp95= 205%→190%).



