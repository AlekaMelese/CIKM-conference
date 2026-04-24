import gradio as gr
import pandas as pd
import os
import html

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "annotation_sheet.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "annotated")
GUIDELINES_PATH = os.path.join(BASE_DIR, "annotation_guidelines.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
        guidelines_text = f.read()
except Exception:
    guidelines_text = "Guidelines file not found."

base_df = pd.read_csv(DATA_PATH)


def get_user_file(annotator_name):
    safe_name = "".join(c for c in annotator_name if c.isalnum() or c in ('_', '-')).strip()
    if not safe_name:
        safe_name = "default"
    return os.path.join(OUTPUT_DIR, f"annotated_sheet_{safe_name}.csv")


def get_user_df(annotator_name):
    path = get_user_file(annotator_name)
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = base_df.copy()
    # Ensure string columns have correct dtype to avoid FutureWarnings
    for col in ['factual_grounding', 'grounding_confidence', 'error_type',
                'clinical_severity', 'free_text_notes', 'doc_completeness',
                'clarity', 'clinical_utility']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)
    return df


def format_source_note(text):
    """Format source clinical note with colored bold section headers using HTML."""
    if not isinstance(text, str) or not text.strip():
        return '<div style="padding:12px; font-family:sans-serif;">No content available</div>'
    import re as _re

    text = html.escape(text.strip())

    # Section headers to highlight (longest first to avoid partial matches)
    source_headers = [
        'Discharge Physical examination',
        'Discharge instructions',
        'Discharge medications',
        'Discharge disposition',
        'Discharge diagnosis',
        'Discharge condition',
        'Followup instructions',
        'Follow-up instructions',
        'Medications on admission',
        'Brief hospital course',
        'History of present illness',
        'Past medical history',
        'Past surgical history',
        'Physical examination',
        'Pertinent results',
        'Chief complaint',
        'Attending doctor',
        'Major procedure',
        'Social history',
        'Family history',
        'Allergies',
    ]

    # Abbreviation-style headers
    abbrev_headers = [
        'PMHx', 'PMH', 'PSHx', 'PSH', 'HPI', 'CC', 'ROS',
        'VITALS', 'Vitals', 'VS',
        'LABS', 'Labs',
        'EXAM', 'Exam',
        'IMAGING', 'Imaging',
    ]

    style = 'color:#1a56db; font-weight:bold;'

    for hdr in source_headers:
        text = _re.sub(
            rf'({_re.escape(hdr)}\s*:)',
            rf'<span style="{style}">\1</span>',
            text,
            flags=_re.IGNORECASE
        )

    for hdr in abbrev_headers:
        text = _re.sub(
            rf'(?<![A-Za-z])({_re.escape(hdr)}\s*:)',
            rf'<span style="{style}">\1</span>',
            text,
        )

    return (
        '<div style="border:1px solid #e5e7eb; padding:16px; border-radius:8px; '
        'background:white; font-family:sans-serif; white-space:pre-wrap; '
        'line-height:1.6; font-size:14px;">'
        f'{text}</div>'
    )


def format_clinical_text(text):
    """Format clinical text for Markdown display."""
    if not isinstance(text, str) or not text.strip():
        return "No content available"
    return text.strip()


def get_sample(idx_val, annotator_name):
    idx = int(idx_val)
    df = get_user_df(annotator_name)
    empty = "No content available"
    if idx >= len(df):
        return ("All annotations complete! Thank you!",
                empty, empty,
                None, None, None, None, "", None, None, None,
                gr.update(visible=False))
    row = df.iloc[idx]

    done = df['factual_grounding'] != ''
    progress = (
        f"**Progress:** {done.sum()} / {len(df)} completed | "
        f"**Current:** {idx + 1} / {len(df)} | "
        f"**Sample ID:** `{row['annotation_id']}` | "
        f"**Annotator:** `{annotator_name}`"
    )

    source_text = format_source_note(row.get('source_note', ''))
    generated_text = format_clinical_text(row.get('generated_summary', ''))

    # Load existing answers (all columns are strings now)
    ans_grounding = row['factual_grounding'] if row['factual_grounding'] else None
    ans_conf = row['grounding_confidence'] if row['grounding_confidence'] else None
    ans_error = row['error_type'] if row['error_type'] else None
    ans_severity = row['clinical_severity'] if row['clinical_severity'] else None
    ans_notes = row.get('free_text_notes', '')
    ans_doc = row.get('doc_completeness', None)
    if not ans_doc:
        ans_doc = None
    ans_clarity = row.get('clarity', None)
    if not ans_clarity:
        ans_clarity = None
    ans_utility = row.get('clinical_utility', None)
    if not ans_utility:
        ans_utility = None

    return (progress, source_text, generated_text,
            ans_grounding, ans_conf, ans_error, ans_severity, ans_notes, ans_doc,
            ans_clarity, ans_utility,
            gr.update(visible=True))


def submit(idx_val, annotator_name, grounding, conf, error_type, severity, notes, doc_comp, clarity, clinical_utility):
    idx = int(idx_val)
    df = get_user_df(annotator_name)
    if idx >= len(df):
        return get_sample(idx, annotator_name) + (idx,)

    if not (grounding and conf and error_type and severity and clarity and clinical_utility):
        gr.Warning("Please answer all required fields (1-6).")
        result = get_sample(idx, annotator_name)
        return (result[0], result[1], result[2],
                grounding, conf, error_type, severity, notes, doc_comp,
                clarity, clinical_utility, result[11], idx)

    if grounding == "Unsupported" and error_type in ["None", None]:
        gr.Warning("Please select an error type for Unsupported.")
        result = get_sample(idx, annotator_name)
        return (result[0], result[1], result[2],
                grounding, conf, error_type, severity, notes, doc_comp,
                clarity, clinical_utility, result[11], idx)

    df.at[idx, 'factual_grounding'] = grounding
    df.at[idx, 'grounding_confidence'] = conf
    df.at[idx, 'error_type'] = error_type
    df.at[idx, 'clinical_severity'] = severity
    df.at[idx, 'free_text_notes'] = notes
    if 'doc_completeness' not in df.columns:
        df['doc_completeness'] = ''
    df.at[idx, 'doc_completeness'] = doc_comp if doc_comp else ''
    df.at[idx, 'clarity'] = clarity if clarity else ''
    df.at[idx, 'clinical_utility'] = clinical_utility if clinical_utility else ''

    df.to_csv(get_user_file(annotator_name), index=False)

    idx += 1
    return get_sample(idx, annotator_name) + (idx,)


def prev_sample(idx_val, annotator_name):
    idx = int(idx_val)
    if idx > 0:
        idx -= 1
    return get_sample(idx, annotator_name) + (idx,)


def start_session(name):
    if not name or not name.strip():
        gr.Warning("Please enter a name or ID.")
        empty = "No content available"
        return (gr.update(visible=True), gr.update(visible=False),
                0, "", "", empty, empty,
                None, None, None, None, "", None, None, None,
                gr.update(visible=True))

    name = name.strip()
    df = get_user_df(name)
    unannotated = df[df['factual_grounding'] == ''].index
    start_idx = int(unannotated[0]) if len(unannotated) > 0 else 0

    result = get_sample(start_idx, name)
    return (gr.update(visible=False), gr.update(visible=True),
            start_idx, name, *result)


# === Build UI ===
CUSTOM_CSS = """
.prose, .prose p, .prose li, .prose strong, .markdown-text, .markdown-text p {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
    font-variant: normal !important;
    font-feature-settings: normal !important;
}
"""

with gr.Blocks(title="Clinical Validation Annotator", css=CUSTOM_CSS) as demo:
    current_idx = gr.State(value=0)
    annotator_id = gr.State(value="")

    # === Welcome Page ===
    with gr.Column(visible=True) as welcome_page:
        gr.Markdown("# Clinical Validation of Factual Grounding in LLM-Generated Discharge Summaries")
        gr.Markdown(
            "### Evaluate whether LLM-generated discharge summary sections "
            "are factually grounded in source clinical notes."
        )
        gr.Markdown("### Please carefully review the Annotation Guidelines before proceeding.")
        gr.Markdown("---")
        with gr.Accordion("Annotation Guidelines (click to expand)", open=False):
            gr.Markdown(f"```text\n{guidelines_text}\n```")
        gr.Markdown("---")
        annotator_input = gr.Textbox(
            label="Enter Your Name / Annotator ID / String with number:",
            placeholder="e.g., Annotator_1, A7X2"
        )
        start_btn = gr.Button("Begin Annotation", variant="primary", size="lg")

    # === Annotation Page ===
    with gr.Column(visible=False) as annotation_page:
        gr.Markdown("# Clinical Validation of Factual Grounding in LLM-Generated Discharge Summaries")
        with gr.Accordion("Annotation Guidelines", open=False):
            gr.Markdown(f"```text\n{guidelines_text}\n```")

        progress_text = gr.Markdown()

        with gr.Column(visible=True) as eval_section:
            gr.Markdown("---")

            gr.Markdown(
                "*Tip: Source notes are written by doctors who may not always use clear section headers. "
                "Some clinical information (e.g., follow-up plans, medication changes, diagnoses) may appear "
                "without a labeled section — it could be embedded within 'Brief hospital course', "
                "'Discharge instructions', or written as free text without any header at all. "
                "Use Ctrl+F / Cmd+F to search for specific terms.*"
            )

            # Source and Generated side by side
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Source Clinical Note (ground truth)")
                    source_display = gr.HTML()
                with gr.Column(scale=1):
                    gr.Markdown("### Generated Summary (evaluate this)")
                    generated_display = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("### Evaluation")

            with gr.Row():
                with gr.Column():
                    grounding = gr.Radio(
                        ["Fully Supported", "Partially Supported", "Unsupported", "Cannot Determine"],
                        label="1. Factual Grounding *",
                        info="Fully Supported=all facts match source, Partially Supported=some facts wrong/missing, Unsupported=contradicts source, Cannot Determine=not enough info to judge"
                    )
                    confidence = gr.Dropdown(
                        ["1", "2", "3", "4", "5"],
                        label="2. Grounding Confidence *",
                        info="1=Very uncertain, 5=Very confident in your grounding judgment"
                    )
                with gr.Column():
                    error_type = gr.Radio(
                        ["None", "Fabrication", "Inaccuracy", "Omission"],
                        label="3. Error Type *",
                        info="None=no error found, Fabrication=info invented/not in source, Inaccuracy=info from source but wrong/altered, Omission=critical info dropped from summary"
                    )
                    severity = gr.Radio(
                        ["None", "Minor", "Major", "Critical"],
                        label="4. Clinical Severity *",
                        info="None=no error, Minor=won't affect care, Major=clinically significant, Critical=could harm patient"
                    )

            with gr.Row():
                clarity = gr.Dropdown(
                    ["1", "2", "3", "4", "5"],
                    label="5. Clarity *",
                    info="1=Very poor, 5=Excellent — How clear and well-organized is the summary?"
                )
                clinical_utility = gr.Dropdown(
                    ["1", "2", "3", "4", "5"],
                    label="6. Clinical Utility *",
                    info="1=Not usable, 5=Ready to use — Would a clinician use this for handoff?"
                )

            doc_completeness = gr.Radio(
                ["N/A", "Sufficient", "Crucial Omission"],
                label="7. Document-Level Completeness (Optional)",
                info="Does the generated summary miss critical information from the source?",
                value="N/A"
            )

            notes = gr.Textbox(
                label="8. Free-Text Notes (Optional)",
                placeholder="What specific error did you identify?"
            )

            with gr.Row():
                prev_btn = gr.Button("Previous Sample", size="lg")
                submit_btn = gr.Button("Submit & Next", variant="primary", size="lg")

    # === Auto-select logic ===
    def on_grounding_change(val):
        """Auto-select based on grounding choice."""
        if val == "Fully Supported":
            # Auto-set Error=None, Severity=None, show all error choices
            return gr.update(value="None", choices=["None", "Fabrication", "Inaccuracy", "Omission"]), gr.update(value="None")
        elif val == "Unsupported":
            # Remove "None" from error choices — error is required
            return gr.update(value=None, choices=["Fabrication", "Inaccuracy", "Omission"]), gr.update()
        elif val == "Partially Supported":
            # Keep all choices including None
            return gr.update(choices=["None", "Fabrication", "Inaccuracy", "Omission"]), gr.update()
        else:
            # Cannot Determine — show all
            return gr.update(choices=["None", "Fabrication", "Inaccuracy", "Omission"]), gr.update()

    def on_error_change(val):
        """Error Type=None → auto-set Grounding=Fully Supported and Severity=None"""
        if val == "None":
            return gr.update(value="Fully Supported"), gr.update(value="None")
        return gr.update(), gr.update()

    grounding.change(on_grounding_change, inputs=[grounding], outputs=[error_type, severity])
    error_type.change(on_error_change, inputs=[error_type], outputs=[grounding, severity])

    # === Event Handlers ===
    all_outputs = [
        progress_text, source_display, generated_display,
        grounding, confidence, error_type, severity, notes, doc_completeness,
        clarity, clinical_utility,
        eval_section,
        current_idx
    ]

    start_btn.click(
        start_session,
        inputs=[annotator_input],
        outputs=[
            welcome_page, annotation_page, current_idx, annotator_id,
            progress_text, source_display, generated_display,
            grounding, confidence, error_type, severity, notes, doc_completeness,
            clarity, clinical_utility,
            eval_section
        ],
        show_progress="hidden"
    )

    submit_btn.click(
        submit,
        inputs=[current_idx, annotator_id, grounding, confidence,
                error_type, severity, notes, doc_completeness,
                clarity, clinical_utility],
        outputs=all_outputs,
        show_progress="hidden"
    )

    prev_btn.click(
        prev_sample,
        inputs=[current_idx, annotator_id],
        outputs=all_outputs,
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())
