# Ethics and Data Use

## Data source

All clinical text used in this study comes from **MIMIC-IV** (Medical Information Mart for Intensive Care, v2.2+), a de-identified publicly available database of electronic health records from Beth Israel Deaconess Medical Center. MIMIC-IV is hosted on PhysioNet and released under the PhysioNet Credentialed Health Data License.

## Compliance

- The primary researcher has completed the required CITI "Data or Specimens Only Research" training and has signed the PhysioNet Data Use Agreement (DUA).
- No protected health information (PHI) is present in the MIMIC-IV release. Dates, physician names, patient names, identifiers, and free-text mentions of these were removed by the dataset maintainers before release.
- Annotators received only the de-identified clinical text. No re-identification was attempted.
- All annotators in the human evaluation study are named collaborators affiliated with the authors' institution: three medical students and one senior postdoctoral clinician at a collaborating university hospital. Each was informed that the material is derived from MIMIC-IV and was given access only to the 100-sample evaluation set.
- The annotation interface was deployed on a Hugging Face Space; the Space link was shared privately with the four annotators and was not advertised publicly.

## What this repository does NOT contain

To remain compliant with the DUA, the following are deliberately excluded:

- `annotation_sheet.csv` and `annotation_key.csv` — contain MIMIC-IV source notes and model outputs derived from them.
- `annotated/*.csv` — completed annotations (kept privately).
- Any raw MIMIC-IV exports.

## What this repository does contain

- Annotation guidelines, rubric, and interface source code.
- Schema of the CSV files, so readers can reproduce the pipeline with their own MIMIC-IV export.
- The inter-annotator agreement analysis script.

## Reproducing the study

To reproduce the annotation pipeline, the reader must:

1. Obtain credentialed access to MIMIC-IV via PhysioNet.
2. Run the fine-tuning and RAG pipeline described in the main paper to produce 200 generated summaries per model × method combination.
3. Run `prepare_annotation_samples.py` (available on request) on those summaries to generate the stratified 100-sample sheet.
4. Launch `ui/app.py` and distribute the link to consented annotators.

## Contact

For questions about the annotation protocol or to request the sampling script, please contact the corresponding author listed in the paper.
