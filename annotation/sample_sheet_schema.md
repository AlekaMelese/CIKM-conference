# Sample Sheet Schema

This document describes the CSV columns used in the annotation interface. The actual CSV is **not** included because it contains MIMIC-IV source notes, which fall under the PhysioNet Data Use Agreement (see [ETHICS.md](ETHICS.md)).

## `annotation_sheet.csv` — 100 rows (one per summary)

Shown to annotators. Contains the content they evaluate and empty columns they fill in.

| Column | Type | Filled by | Description |
|--------|------|-----------|-------------|
| `annotation_id` | string | sampling script | Stable identifier, e.g. `CVAL_0001` |
| `generated_summary` | text | sampling script | Model output being evaluated |
| `target_summary` | text | sampling script | Reference discharge summary (not shown during annotation; kept for later analysis) |
| `source_note` | text | sampling script | De-identified MIMIC-IV clinical note |
| `factual_grounding` | enum | annotator | Fully Supported / Partially Supported / Unsupported / Cannot Determine |
| `grounding_confidence` | int 1–5 | annotator | Annotator's confidence in field above |
| `error_type` | enum | annotator | None / Fabrication / Inaccuracy / Omission |
| `clinical_severity` | enum | annotator | None / Minor / Major / Critical |
| `free_text_notes` | text | annotator | Optional observations |
| `doc_completeness` | enum | annotator | N/A / Sufficient / Crucial Omission |
| `clarity` | int 1–5 | annotator | Likert rating |
| `clinical_utility` | int 1–5 | annotator | Likert rating |

## `annotation_key.csv` — 100 rows (one per summary)

**Private — not shown to annotators.** Contains the metadata needed to join annotation results back to model identity and automatic scores.

| Column | Type | Description |
|--------|------|-------------|
| `annotation_id` | string | Matches `annotation_sheet.csv` |
| `_model` | string | One of LLaMA, Gemma, Phi, Qwen |
| `_method` | string | `ft` (fine-tuned) or `hybrid` (Structure-Only RAG) |
| `_note_id` | string | Source note identifier (MIMIC-IV-derived) |
| `_auto_score` | float | S-PubMedBERT factual-consistency score |
| `_auto_label` | string | Discretised score: SUPPORTED / PARTIAL / UNSUPPORTED |
| `_hallucination_rate` | float | Proportion of sentences flagged as low-grounding |
| `_grounded_sents` | int | Count of grounded sentences |
| `_total_sents` | int | Total sentence count |

## Annotator output — `annotated_sheet_<name>.csv`

Each annotator produces one file matching `annotation_sheet.csv` with the six required fields filled in. The [`compute_agreement.py`](compute_agreement.py) script expects these files in the `annotated/` directory.

Filename format: `annotated_sheet_<AnnotatorID>.csv` where `AnnotatorID` is alphanumeric (underscores and hyphens allowed).
