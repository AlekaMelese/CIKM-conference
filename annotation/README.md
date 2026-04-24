# Clinical Validation Study

This folder documents the human evaluation conducted for the accompanying paper on hybrid PEFT with retrieval-augmented generation for clinical discharge summaries. It contains the annotation protocol, the rubric, the interface code, and the agreement-analysis script used to compute inter-annotator reliability.

The actual MIMIC-IV clinical text is **not** included in this repository, in line with the PhysioNet Data Use Agreement (see [ETHICS.md](ETHICS.md)).

---

## Study at a glance

| Aspect | Value |
|--------|-------|
| Samples | 100 generated discharge summaries from the held-out test set |
| Annotators | 3 medical students + 1 senior postdoctoral clinician at a collaborating university hospital |
| Models covered | LLaMA-3.1-8B, Gemma-2-9B, Phi-3-Medium-14B, Qwen2.5-7B |
| Methods covered | FT-only and Hybrid (Structure-Only RAG) |
| Rubric fields per sample | 8 (4 categorical + 3 Likert + 1 free-text) |
| Blinding | Model identity, method, and automatic label hidden |
| Interface | Gradio web app, hosted on Hugging Face Spaces |
| Agreement metrics | Fleiss' kappa, pairwise Cohen's kappa, ICC(2,1) |
| Pilot | 10 samples reviewed by the advisor before main annotation |

---

## Files

| File | Content |
|------|---------|
| [`guidelines.txt`](guidelines.txt) | Full annotation guidelines shown to annotators |
| [`rubric.md`](rubric.md) | Tabular summary of the 8 evaluation fields |
| [`sample_sheet_schema.md`](sample_sheet_schema.md) | CSV column specification (no MIMIC-IV data) |
| [`ui/app.py`](ui/app.py) | Gradio annotation interface source code |
| [`ui/requirements.txt`](ui/requirements.txt) | Python dependencies for the UI |
| [`ui/screenshots/`](ui/screenshots/) | Reference screenshots of the interface |
| [`compute_agreement.py`](compute_agreement.py) | Script computing Fleiss' kappa, Cohen's kappa, ICC, and Spearman correlation with the automatic factuality score |
| [`ETHICS.md`](ETHICS.md) | MIMIC-IV DUA compliance statement |

---

## Sampling

The 100 source notes were drawn from the held-out test set with stratification across the four transformer architectures and the two generation configurations (FT-only and Hybrid). Within each stratum, samples were further balanced across the three automatic S-PubMedBERT factuality labels (SUPPORTED, PARTIAL, UNSUPPORTED) so that the full range of grounding levels is represented.

Filters applied before sampling:
- Malformed or repetitive outputs (repetition regex `(.{2,30})\1{4,}`) were removed.
- Duplicate section headers were collapsed.
- Only summaries containing all 11 canonical sections were retained.

## Blinding

Annotators did not know the generating model, the generation method (FT-only vs. Hybrid), or the automatic label assigned by the factuality score. Pairs of summaries for the same source note were randomised independently.

## Agreement analysis

The [`compute_agreement.py`](compute_agreement.py) script reports, for each categorical rubric field:
- Fleiss' kappa across the three medical-student annotators (primary).
- Pairwise Cohen's kappa between every annotator pair (secondary).
- ICC(2,1) for the three Likert-scale fields (Grounding Confidence, Clarity, Clinical Utility).

It also computes Spearman rank correlation between the automatic S-PubMedBERT score and the majority-vote Factual Grounding label, to validate the automatic metric.

Run with:
```bash
python compute_agreement.py
```
after all annotators deposit their `annotated_sheet_<name>.csv` files in the expected directory.

---

## Paper reference

See Section *Human/Clinical Validation* of the paper for the full methodology description.
