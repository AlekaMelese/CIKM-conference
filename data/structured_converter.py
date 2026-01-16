#!/usr/bin/env python3
"""
Structured Discharge Summary Converter

Converts raw MIMIC-IV clinical notes into structured 11-section format
for fine-tuning clinical summarization models.

Sections:
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
"""

import re
import pandas as pd
from typing import Tuple, Optional


class StructuredConverter:
    """Converts clinical notes to structured 11-section discharge summaries"""

    def __init__(self):
        self.sections = [
            "Case Type",
            "Patient & Service",
            "Chief Complaint / Admission Context",
            "History of Present Illness (HPI)",
            "Past Medical / Surgical History",
            "Medications (Discharge / Ongoing)",
            "Physical Examination (summarized)",
            "Investigations / Labs / Imaging",
            "Assessment / Impression",
            "Discharge Condition",
            "Follow-Up & Recommendations"
        ]

    def extract_case_type(self, input_text: str) -> str:
        """Extract case type from clinical note"""
        # Try to find service/specialty
        service_match = re.search(r'service[:\s]+(\w+)', input_text, re.IGNORECASE)
        if service_match:
            return service_match.group(1).strip()

        # Try diagnosis-based case type
        diag_match = re.search(r'discharge diagnosis[:\s]+([^\n]+)', input_text, re.IGNORECASE)
        if diag_match:
            return diag_match.group(1).strip()[:100]

        return "Medical admission"

    def extract_patient_service(self, input_text: str) -> str:
        """Extract patient demographics and service"""
        parts = []

        # Sex
        sex_match = re.search(r'sex[:\s]+([MF])', input_text, re.IGNORECASE)
        if sex_match:
            parts.append(f"{'Male' if sex_match.group(1).upper() == 'M' else 'Female'} patient")

        # Service
        service_match = re.search(r'service[:\s]+(\w+)', input_text, re.IGNORECASE)
        if service_match:
            parts.append(service_match.group(1).upper())

        return ", ".join(parts) if parts else "Patient admitted for care"

    def extract_chief_complaint(self, input_text: str, target_text: str) -> str:
        """Extract chief complaint"""
        combined = input_text + " " + target_text

        patterns = [
            r'chief complaint[:\s]+([^\n]+)',
            r'reason for (?:admission|visit)[:\s]+([^\n]+)',
            r'admitted (?:for|with)[:\s]+([^\n]+)',
            r'presents? with[:\s]+([^.\n]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                cc = match.group(1).strip()
                if 10 < len(cc) < 300:
                    return cc

        # Extract from major procedure
        proc_match = re.search(r'major procedure[:\s]+([^\n]+)', combined, re.IGNORECASE)
        if proc_match:
            procedure = proc_match.group(1).strip()
            if procedure.lower() not in ['none', 'n/a', 'na', '']:
                return procedure

        return "Patient admitted for evaluation and management."

    def extract_hpi(self, input_text: str, target_text: str) -> str:
        """Extract history of present illness"""
        hpi_match = re.search(
            r'history of present illness[:\s]+(.*?)(?=past medical|social history|medications|physical exam|pertinent|$)',
            input_text,
            re.IGNORECASE | re.DOTALL
        )

        if hpi_match:
            hpi = hpi_match.group(1).strip()
            hpi = re.sub(r'\s+', ' ', hpi)
            if len(hpi) > 100:
                return hpi[:1000] + '...' if len(hpi) > 1000 else hpi

        # Use narrative target as HPI
        if len(target_text) > 100:
            narrative = re.sub(r'\s+', ' ', target_text.strip())
            sections = re.split(r'\n\s*(?:[A-Z][a-z]+\s*-|Neuro\s*-|Pulmonary\s*-|GI\s*-|CV\s*-)', narrative)
            first_section = sections[0].strip()
            if len(first_section) > 50:
                return first_section[:1200] + '...' if len(first_section) > 1200 else first_section

        return "Clinical course documented in the medical record."

    def extract_pmh(self, input_text: str) -> str:
        """Extract past medical/surgical history"""
        pmh_match = re.search(
            r'past medical history[:\s]+(.*?)(?=past surgical|social history|family history|medications|physical exam|allergies|$)',
            input_text,
            re.IGNORECASE | re.DOTALL
        )

        if pmh_match:
            pmh = pmh_match.group(1).strip()
            pmh = re.sub(r'\s+', ' ', pmh)
            if pmh.lower() not in ['none', 'n/a', 'na', 'not available', 'see above']:
                if len(pmh) > 30:
                    return pmh[:600] if len(pmh) > 600 else pmh

        # Try past surgical history
        psh_match = re.search(
            r'past surgical history[:\s]+(.*?)(?=social history|family history|medications|$)',
            input_text,
            re.IGNORECASE | re.DOTALL
        )
        if psh_match:
            psh = psh_match.group(1).strip()
            if len(psh) > 20 and psh.lower() not in ['none', 'n/a']:
                return f"Past Surgical History: {psh[:400]}"

        return "No significant past medical history documented."

    def extract_medications(self, input_text: str, target_text: str) -> Tuple[str, str]:
        """Extract discharge and ongoing medications"""
        combined = input_text + " " + target_text

        discharge_patterns = [
            r'discharge medications[:\s]+(.*?)(?=discharge disposition|discharge diagnosis|discharge condition|follow|$)',
            r'medications on discharge[:\s]+(.*?)(?=discharge disposition|discharge diagnosis|$)',
        ]

        discharge_meds = None
        for pattern in discharge_patterns:
            match = re.search(pattern, combined, re.IGNORECASE | re.DOTALL)
            if match:
                meds = re.sub(r'\s+', ' ', match.group(1).strip())
                if len(meds) > 30 and meds.lower() not in ['none', 'n/a', 'see above']:
                    discharge_meds = meds[:600] if len(meds) > 600 else meds
                    break

        if not discharge_meds:
            adm_match = re.search(
                r'medications on admission[:\s]+(.*?)(?=discharge medications|physical exam|allergies|$)',
                input_text,
                re.IGNORECASE | re.DOTALL
            )
            if adm_match:
                meds = adm_match.group(1).strip()
                if len(meds) > 30:
                    discharge_meds = f"Continue home medications: {meds[:500]}"

        if not discharge_meds:
            discharge_meds = "Discharge medications as prescribed by treating physician."

        ongoing_meds = "Continue home medications as previously prescribed."

        return discharge_meds, ongoing_meds

    def extract_physical_exam(self, input_text: str) -> str:
        """Extract physical examination findings"""
        pe_match = re.search(
            r'physical exam(?:ination)?[:\s]+(.*?)(?=pertinent results?|labs?|imaging|medications|social history|$)',
            input_text,
            re.IGNORECASE | re.DOTALL
        )

        if pe_match:
            pe = re.sub(r'\s+', ' ', pe_match.group(1).strip())
            if pe.lower() not in ['see above', 'wnl', 'normal'] and len(pe) > 50:
                return pe[:800] if len(pe) > 800 else pe

        # Look for vital signs
        vitals_match = re.search(r'(?:vitals?|vital signs?)[:\s]+(.*?)(?:\n|$)', input_text, re.IGNORECASE)
        if vitals_match:
            return f"Physical examination documented. Vitals: {vitals_match.group(1).strip()}"

        return "Physical examination findings documented."

    def extract_investigations(self, input_text: str) -> str:
        """Extract laboratory and imaging results"""
        labs_match = re.search(
            r'pertinent results?[:\s]+(.*?)(?=medications|discharge|social history|$)',
            input_text,
            re.IGNORECASE | re.DOTALL
        )

        if labs_match:
            labs = re.sub(r'\s+', ' ', labs_match.group(1).strip())
            if len(labs) > 50:
                return labs[:800] if len(labs) > 800 else labs

        imaging_match = re.search(
            r'(?:imaging|radiology|studies?)[:\s]+(.*?)(?=medications|discharge|$)',
            input_text,
            re.IGNORECASE | re.DOTALL
        )
        if imaging_match:
            img = imaging_match.group(1).strip()
            if len(img) > 30:
                return f"Imaging: {img[:600]}"

        return "Laboratory and imaging results as documented."

    def extract_assessment(self, input_text: str, target_text: str) -> str:
        """Extract clinical assessment/impression"""
        combined = input_text + " " + target_text

        patterns = [
            r'discharge diagnosis[:\s]+(.*?)(?=discharge condition|discharge disposition|medications|follow)',
            r'discharge diagnos[ei]s[:\s]+(.*?)(?=\n\n|discharge|medications)',
            r'(?:final )?diagnosis[:\s]+(.*?)(?=discharge condition|medications|follow)',
            r'assessment[:\s]+([^\n]{20,400})',
            r'impression[:\s]+([^\n]{20,400})',
        ]

        for pattern in patterns:
            match = re.search(pattern, combined, re.IGNORECASE | re.DOTALL)
            if match:
                assessment = re.sub(r'\s+', ' ', match.group(1).strip())
                assessment = re.sub(r'^(Primary|Secondary|Principal):\s*', '', assessment, flags=re.IGNORECASE)
                if len(assessment) > 20:
                    assessment = assessment.split('Discharge condition')[0]
                    assessment = assessment.split('Discharge disposition')[0].strip()
                    if len(assessment) > 20:
                        return assessment[:500] if len(assessment) > 500 else assessment

        cc_match = re.search(r'chief complaint[:\s]+([^\n]{20,200})', combined, re.IGNORECASE)
        if cc_match:
            return f"Patient admitted for {cc_match.group(1).strip()}"

        return "See clinical documentation for complete assessment."

    def extract_discharge_condition(self, input_text: str) -> str:
        """Extract discharge condition"""
        dc_match = re.search(
            r'discharge condition[:\s]+(.*?)(?=discharge disposition|discharge instructions|follow|$)',
            input_text,
            re.IGNORECASE | re.DOTALL
        )

        if dc_match:
            condition = re.sub(r'\s+', ' ', dc_match.group(1).strip())
            if len(condition) > 10:
                return condition[:300] if len(condition) > 300 else condition

        return "Stable condition at discharge."

    def extract_followup(self, input_text: str, target_text: str) -> str:
        """Extract follow-up instructions"""
        combined = input_text + " " + target_text

        followup_patterns = [
            r'follow[\s-]?up (?:instructions?|appointments?)[:\s]+(.*?)(?=discharge disposition|$)',
            r'discharge instructions?[:\s]+(.*?)(?=discharge disposition|$)',
        ]

        for pattern in followup_patterns:
            match = re.search(pattern, combined, re.IGNORECASE | re.DOTALL)
            if match:
                followup = re.sub(r'\s+', ' ', match.group(1).strip())
                if len(followup) > 30:
                    return followup[:600] if len(followup) > 600 else followup

        return "Follow up with primary care physician and specialists as directed."

    def generate_structured_summary(self, note_id: str, input_text: str, target_text: str) -> str:
        """Generate complete structured summary"""
        case_type = self.extract_case_type(input_text)
        patient_service = self.extract_patient_service(input_text)
        chief_complaint = self.extract_chief_complaint(input_text, target_text)
        hpi = self.extract_hpi(input_text, target_text)
        pmh = self.extract_pmh(input_text)
        discharge_meds, ongoing_meds = self.extract_medications(input_text, target_text)
        physical_exam = self.extract_physical_exam(input_text)
        investigations = self.extract_investigations(input_text)
        assessment = self.extract_assessment(input_text, target_text)
        discharge_condition = self.extract_discharge_condition(input_text)
        followup = self.extract_followup(input_text, target_text)

        structured = f"""Case Type: {case_type}

• Patient & Service: {patient_service}

• Chief Complaint / Admission Context: {chief_complaint}

• History of Present Illness (HPI): {hpi}

• Past Medical / Surgical History: {pmh}

• Medications (Discharge / Ongoing):
  Discharge: {discharge_meds}
  Ongoing: {ongoing_meds}

• Physical Examination (summarized): {physical_exam}

• Investigations / Labs / Imaging: {investigations}

• Assessment / Impression: {assessment}

• Discharge Condition: {discharge_condition}

• Follow-Up & Recommendations: {followup}"""

        return structured


def convert_dataset(input_csv: str, output_csv: str):
    """Convert MIMIC-IV dataset to structured format

    Args:
        input_csv: Path to input CSV with 'note_id', 'input', 'target' columns
        output_csv: Path to save output CSV with additional 'structured_target' column
    """
    print("=" * 80)
    print("STRUCTURED DISCHARGE SUMMARY CONVERTER")
    print("=" * 80)
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print("=" * 80)

    # Load dataset
    print(f"\n📂 Loading dataset...")
    df = pd.read_csv(input_csv)
    print(f"   ✓ Loaded {len(df)} records")
    print(f"   ✓ Columns: {df.columns.tolist()}")

    # Initialize converter
    converter = StructuredConverter()

    # Process each record
    structured_summaries = []
    print(f"\n🔄 Converting to structured format...")

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"   Progress: {idx}/{len(df)} records ({100*idx/len(df):.1f}%)")

        note_id = str(row['note_id'])
        input_text = str(row['input']) if pd.notna(row['input']) else ""
        target_text = str(row['target']) if pd.notna(row['target']) else ""

        structured = converter.generate_structured_summary(note_id, input_text, target_text)
        structured_summaries.append(structured)

    print(f"   Progress: {len(df)}/{len(df)} records (100.0%)")

    # Add new column
    df['structured_target'] = structured_summaries

    # Save result
    print(f"\n💾 Saving structured dataset...")
    df.to_csv(output_csv, index=False)

    print(f"\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"📊 Statistics:")
    print(f"   Total records:      {len(df)}")
    print(f"   New column added:   'structured_target'")
    print(f"   Output file:        {output_csv}")

    # Show sample
    print(f"\n" + "=" * 80)
    print("SAMPLE STRUCTURED SUMMARY:")
    print("=" * 80)
    print(structured_summaries[0][:1500])
    print("=" * 80)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert MIMIC-IV notes to structured format')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')

    args = parser.parse_args()
    convert_dataset(args.input, args.output)
