# Supplementary Materials

This folder contains explainability output files from Phase 3 of the Hybrid PEFT+RAG framework, generated using the Phi-3-Medium-14B model on 200 test samples.

> **Note**: Some JSON files exceed GitHub's display limit (~1 MB) but can be downloaded or cloned normally.

## File Descriptions

### 1. confidence_scores.json (91 KB)
Multi-factor confidence assessment for each generated summary.

**Structure:**
```json
{
  "note_id": "12601669-DS-21",
  "overall_confidence": 0.847,
  "confidence_level": "HIGH",
  "component_scores": {
    "retrieval_quality": 0.92,
    "structure_completeness": 1.0,
    "input_output_consistency": 0.78,
    "length_appropriateness": 0.85,
    "entity_preservation": 0.81
  }
}
```

### 2. factual_alignment.json (3.5 MB)
Sentence-level factual verification using S-PubMedBERT embeddings.

**Structure:**
```json
{
  "note_id": "12601669-DS-21",
  "sentence_alignments": [
    {
      "generated_sentence": "Patient presents with a cystocele...",
      "alignment_score": 0.967,
      "supporting_evidence": [
        {
          "sentence": "History of present illness: The patient presents with a cystocele...",
          "similarity": 0.967
        }
      ],
      "factual_status": "SUPPORTED"
    }
  ]
}
```

**Factual Status Categories:**
- `SUPPORTED`: similarity ≥ 0.70 (factually grounded)
- `PARTIAL`: similarity 0.50-0.70 (partially supported)
- `UNSUPPORTED`: similarity < 0.50 (potential hallucination)

### 3. evidence_attributions.json (13.6 MB)
Section-level evidence attribution tracing which retrieved cases influenced each generated section.

**Structure:**
```json
{
  "note_id": "12601669-DS-21",
  "retrieved_cases": [
    {"note_id": "11086503-DS-18", "score": 0.9997},
    {"note_id": "10960232-DS-14", "score": 0.9991},
    {"note_id": "13236317-DS-7", "score": 0.9985}
  ],
  "section_attributions": {
    "Case Type": [
      {
        "sentence": "Anterior colporrhaphy cystocele repair",
        "evidence": [
          {
            "text": "colpocleisis, perineorrhaphy, tension free vaginal tape...",
            "doc_id": "10960232-DS-14",
            "retrieval_score": 0.999,
            "similarity": 0.889
          }
        ]
      }
    ],
    "Chief Complaint": [...],
    "History of Present Illness": [...],
    ...
  }
}
```

### 4. counterfactual_explanations.json (201 KB)
Counterfactual analysis examining how output would change if specific conditions were absent.

**Structure:**
```json
{
  "note_id": "12601669-DS-21",
  "counterfactuals": [
    {
      "condition_removed": "hypertension",
      "expected_changes": {
        "assessment": "Would not mention hypertension as diagnosis",
        "medications": ["lisinopril", "amlodipine", "losartan", "hydrochlorothiazide"],
        "past_medical": "Would not list hypertension in medical history"
      }
    }
  ]
}
```

### 5. counterfactual_report.txt (9.5 KB)
Human-readable summary of counterfactual analysis for 15 representative cases.

**Sample:**
```
COUNTERFACTUAL 1: 12601669-DS-21
COUNTERFACTUAL SCENARIO: Patient does NOT have hypertension

Expected Changes:
  - Assessment: Would not mention hypertension as diagnosis
  - Medications: Would not include hypertension-related medications
  - Past_Medical: Would not list hypertension in medical history

CLINICAL REASONING: The model's mention of hypertension is likely driven by:
  1. Condition-specific symptoms/findings in input
  2. Related medications prescribed
  3. Similar cases in retrieved evidence
```

## Usage

To load the JSON files in Python:

```python
import json

# Load factual alignment data
with open('factual_alignment.json', 'r') as f:
    alignments = json.load(f)

# Analyze hallucination rates
for sample in alignments:
    statuses = [s['factual_status'] for s in sample['sentence_alignments']]
    unsupported_rate = statuses.count('UNSUPPORTED') / len(statuses)
    print(f"{sample['note_id']}: {unsupported_rate:.1%} unsupported")
```

## Summary Statistics

| Metric | Value |
|--------|-------|
| Average confidence score | 0.72 |
| High confidence (≥0.70) | 77-83% |
| Evidence traceability | 91% |
| Sections covered | 11 |
