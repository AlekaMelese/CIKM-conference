# Data Instructions

## MIMIC-IV Dataset

This project uses the **MIMIC-IV** (Medical Information Mart for Intensive Care IV) database, which contains de-identified clinical data from Beth Israel Deaconess Medical Center.

### Data Access

Due to the sensitive nature of clinical data and PhysioNet's data use agreement, we **cannot redistribute the raw dataset**. To obtain access:

1. **Create a PhysioNet account**: https://physionet.org/register/

2. **Complete required training**:
   - CITI "Data or Specimens Only Research" course
   - Submit certificate to PhysioNet

3. **Sign the data use agreement**:
   - https://physionet.org/content/mimiciv/

4. **Download the dataset**:
   ```bash
   wget -r -N -c -np --user [USERNAME] --ask-password https://physionet.org/files/mimiciv/2.2/
   ```

### Required Tables

For this project, you need the following MIMIC-IV tables:

- `hosp/discharge.csv` - Discharge summaries
- `hosp/admissions.csv` - Admission information
- `hosp/patients.csv` - Patient demographics

### Preprocessing

After obtaining the data, run the preprocessing scripts:

**Step 1: Extract discharge summaries from MIMIC-IV**
```bash
python data_processing/prepare_dataset.py \
    --mimic_path /path/to/mimiciv/ \
    --output_path data_processing/processed/
```

**Step 2: Convert to structured 11-section format**
```bash
python data_processing/structured_converter.py \
    --input data_processing/processed/discharge_notes.csv \
    --output data_processing/processed/structured_notes.csv
```

The structured converter extracts and organizes clinical notes into 11 sections:
1. Case Type
2. Patient & Service
3. Chief Complaint / Admission Context
4. History of Present Illness (HPI)
5. Past Medical / Surgical History
6. Medications (Discharge / Ongoing)
7. Physical Examination
8. Investigations / Labs / Imaging
9. Assessment / Impression
10. Discharge Condition
11. Follow-Up & Recommendations

**Step 3: Prepare RAG corpus (for Phase 2)**
```bash
python data_processing/prepare_rag_corpus.py \
    --input data_processing/processed/structured_notes.csv \
    --output data_processing/rag_corpus/
```

This will create:
- `data_processing/processed/train.json` (70%)
- `data_processing/processed/val.json` (15%)
- `data_processing/processed/test.json` (15%)
- `data_processing/rag_corpus/rag_corpus.csv` (train + val for retrieval)

### Data Format

Each record contains:

```json
{
    "note_id": "12345678-DS-1",
    "input": "sex:M Service:SURGERY Allergies:... Chief complaint:...",
    "target": "Case Type: ... • Patient & Service: ...",
    "input_tokens": 7331,
    "target_tokens": 2443
}
```

### Statistics

| Metric | Value |
|--------|-------|
| Avg input length | 7,331 chars |
| Avg target length | 2,443 chars |
| Max input tokens | 10,484 |
| Max target tokens | 3,584 |
| Compression ratio | 4.4:1 |

### Medical Specialties Distribution

| Specialty | Percentage |
|-----------|------------|
| MEDICINE | 48.2% |
| SURGERY | 15.3% |
| ORTHOPAEDICS | 10.2% |
| Others | 26.3% |


