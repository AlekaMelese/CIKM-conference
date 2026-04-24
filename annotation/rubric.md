# Annotation Rubric

Annotators evaluated each generated discharge summary along eight fields. Fields 1–6 are required; 7 and 8 are optional. The exact wording shown in the interface is in [`guidelines.txt`](guidelines.txt).

| # | Field | Type | Labels / Scale | Notes |
|---|-------|------|----------------|-------|
| 1 | Factual Grounding | Categorical | Fully Supported, Partially Supported, Unsupported, Cannot Determine | Primary grounding judgment |
| 2 | Grounding Confidence | Likert 1–5 | 1 = very uncertain, 5 = very confident | Confidence in field 1 |
| 3 | Error Type | Categorical | None, Fabrication, Inaccuracy, Omission | Required if field 1 = Unsupported |
| 4 | Clinical Severity | Categorical | None, Minor, Major, Critical | How dangerous the worst error is |
| 5 | Clarity | Likert 1–5 | 1 = very unclear, 5 = very clear | Summary readability |
| 6 | Clinical Utility | Likert 1–5 | 1 = not usable, 5 = ready for handoff | Usability for real clinical workflow |
| 7 | Document-Level Completeness | Categorical (optional) | N/A, Sufficient, Crucial Omission | Whether the summary misses source-note content that would change care |
| 8 | Free-Text Notes | Free text (optional) | — | Observations, uncertainty, flagged entities |

## Consistency rules enforced by the interface

- Selecting **Fully Supported** pre-sets Error Type and Clinical Severity to *None*.
- Selecting **Unsupported** requires Error Type to be non-None.
- Annotators may override any pre-set value at any time.

## Label definitions

### Factual Grounding
- **Fully Supported** — every clinical fact in the summary is supported by the source note.
- **Partially Supported** — at least one clinical fact cannot be verified from the source, but most are supported.
- **Unsupported** — the summary contains fabricated or contradictory clinical content.
- **Cannot Determine** — the source note does not contain enough information to judge.

### Error Type
- **Fabrication** — information absent from the source note is presented as fact.
- **Inaccuracy** — information is present in the source but reported with wrong values, wrong temporal ordering, or wrong entity.
- **Omission** — clinically important content from the source is dropped from the summary.

### Clinical Severity
- **Minor** — incorrect detail unlikely to affect care.
- **Major** — incorrect detail that could influence documentation quality, triage, or follow-up, but would usually be caught.
- **Critical** — incorrect detail that could cause direct patient harm if acted upon.
