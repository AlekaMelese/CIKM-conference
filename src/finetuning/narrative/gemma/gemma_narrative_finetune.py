import torch
from unsloth import FastLanguageModel
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback, TrainerCallback
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import time
import sys
sys.path.insert(0, './Final')
from narrative_plotting import create_narrative_training_plots

# Import evaluation metrics
try:
    import evaluate
    METRICS_AVAILABLE = True
except ImportError:
    print("⚠️ evaluate library not available. Install: pip install evaluate")
    METRICS_AVAILABLE = False


class GemmaNarrativeConfig:
    """Configuration for Gemma-2-9B fine-tuning on NARRATIVE summaries - 5000 samples"""

    # Model settings
    MODEL_NAME = "unsloth/gemma-2-9b-it-bnb-4bit"
    MAX_SEQ_LENGTH = 4096
    DTYPE = None  # Auto-detect
    LOAD_IN_4BIT = True

    # LoRA settings (optimized for Gemma 2)
    LORA_R = 64
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    # Training settings (adjusted for 5000 samples)
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 3
    WARMUP_STEPS = 10
    MAX_GRAD_NORM = 1.0
    WEIGHT_DECAY = 0.05
    LR_SCHEDULER = "linear"
    OPTIMIZER = "adamw_8bit"

    # Early stopping configuration
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 0.001

    # Dataset - 5000 samples (using 'target' column for narrative)
    DATASET_PATH = "./Data/5000_structured.csv"

    # Custom split: Test=200, Val=300, Train=4500
    NUM_TEST_SAMPLES = 200
    NUM_VAL_SAMPLES = 300
    RANDOM_STATE = 42

    # Output paths - NARRATIVE-specific directory
    OUTPUT_DIR = "./Final/Gemma/Narrative/Finetuning/outputs"
    CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
    FINAL_MODEL_DIR = f"{OUTPUT_DIR}/final_model"
    MERGED_MODEL_DIR = f"{OUTPUT_DIR}/merged_model"
    PLOTS_DIR = f"{OUTPUT_DIR}/plots"
    LOGS_DIR = f"{OUTPUT_DIR}/logs"

    # Logging
    LOGGING_STEPS = 10
    SAVE_STEPS = 100
    EVAL_STEPS = 50

    # Phase 2 preparation
    SAVE_METADATA = True
    SAVE_TEST_SET = True


def create_narrative_prompt_with_examples(input_text: str, target_text: str = None,
                                           example_pool: list = None) -> str:
    """
    Create Gemma-2-Instruct format prompt for WELL-ORGANIZED NARRATIVE summary generation

    Emphasizes comprehensive, organized narrative paragraphs (200-400 words typical)
    Uses 'target' column which contains narrative summaries of varying lengths (50-600 words)
    """

    # CLEAN SIMPLE PROMPT: NO examples, NO verbose instructions
    # This prevents the model from learning to echo instructions
    if example_pool is None or len(example_pool) < 2:
        if target_text:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical AI assistant specialized in generating narrative discharge summaries.<|eot_id|><|start_header_id|>user<|end_header_id|>

Write a well-organized narrative discharge summary from the clinical note below. Use flowing paragraphs without bullet points or section headers.

Clinical Note:
{input_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{target_text}<|eot_id|>"""
        else:
            # Inference format: CLEAN and SIMPLE (matches training format)
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical AI assistant specialized in generating narrative discharge summaries.<|eot_id|><|start_header_id|>user<|end_header_id|>

Write a well-organized narrative discharge summary from the clinical note below. Use flowing paragraphs without bullet points or section headers.

Clinical Note:
{input_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt

    # Randomly select 2 examples from the pool
    import random
    examples = random.sample(example_pool, min(2, len(example_pool)))

    if target_text:
        # Training format with real few-shot examples
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical AI assistant specialized in generating comprehensive narrative discharge summaries. Write well-organized, flowing paragraph summaries without bullet points or section headers.<|eot_id|><|start_header_id|>user<|end_header_id|>

Here are two examples of well-organized narrative discharge summaries:

EXAMPLE 1:
Clinical Note: {examples[0]['input'][:1000]}...

Narrative Summary: {examples[0]['target']}

---

EXAMPLE 2:
Clinical Note: {examples[1]['input'][:1000]}...

Narrative Summary: {examples[1]['target']}

---

Now generate a comprehensive, well-organized narrative discharge summary for this new clinical note.

Remember:
- Write in flowing paragraphs (NO bullets, NO headers)
- Organize logically: demographics/admission → hospital course → discharge
- Be thorough and complete
- Adapt length based on case complexity

Clinical Note:
{input_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{target_text}<|eot_id|>"""
    else:
        # Inference format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical AI assistant specialized in generating comprehensive narrative discharge summaries. Write well-organized, flowing paragraph summaries without bullet points or section headers.<|eot_id|><|start_header_id|>user<|end_header_id|>

Here are two examples of well-organized narrative discharge summaries:

EXAMPLE 1:
Clinical Note: {examples[0]['input'][:1000]}...

Narrative Summary: {examples[0]['target']}

---

EXAMPLE 2:
Clinical Note: {examples[1]['input'][:1000]}...

Narrative Summary: {examples[1]['target']}

---

Now generate a comprehensive, well-organized narrative discharge summary for this new clinical note.

Remember:
- Write in flowing paragraphs (NO bullets, NO headers)
- Organize logically: demographics/admission → hospital course → discharge
- Be thorough and complete
- Adapt length based on case complexity

Clinical Note:
{input_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    return prompt


def prepare_dataset(config: GemmaNarrativeConfig):
    """Load and split dataset with custom sizes: Train=4500, Val=300, Test=200

    Uses 'target' column (narrative format) NOT 'structured_target'
    Target lengths vary from ~50 to 600+ words depending on case complexity
    """

    print("="*80)
    print("NARRATIVE DATASET PREPARATION - 5000 SAMPLES")
    print("Using 'target' column (well-organized narrative paragraphs)")
    print("="*80)

    # Load dataset
    df = pd.read_csv(config.DATASET_PATH)
    print(f"\n✓ Loaded dataset: {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")

    # Verify we have required columns
    if 'target' not in df.columns:
        raise ValueError("Dataset must contain 'target' column (narrative summaries)")

    # Analyze target lengths
    df['target_words'] = df['target'].str.split().str.len()

    print(f"\n✓ Data overview:")
    print(f"  Input column: 'input' (clinical notes)")
    print(f"  Target column: 'target' (narrative summaries)")
    print(f"  Avg input chars: {df['input'].str.len().mean():.0f}")
    print(f"  Avg target chars: {df['target'].str.len().mean():.0f}")
    print(f"  Avg target words: {df['target_words'].mean():.0f}")
    print(f"  Target word range: {df['target_words'].min():.0f} - {df['target_words'].max():.0f}")

    # Verify we have 5000 samples
    if len(df) < (config.NUM_TEST_SAMPLES + config.NUM_VAL_SAMPLES):
        raise ValueError(f"Dataset has only {len(df)} samples, need at least {config.NUM_TEST_SAMPLES + config.NUM_VAL_SAMPLES}")

    # Create custom split: Test=200, Val=300, Train=remaining
    # First split off test set (200 samples)
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.NUM_TEST_SAMPLES,
        random_state=config.RANDOM_STATE
    )

    # Then split train_val into train and val (300 samples for val)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.NUM_VAL_SAMPLES,
        random_state=config.RANDOM_STATE
    )

    print(f"\n✓ Custom split sizes:")
    print(f"  Train: {len(train_df)} samples (90%)")
    print(f"  Val:   {len(val_df)} samples (6%)")
    print(f"  Test:  {len(test_df)} samples (4%)")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} samples")

    # Save test set for Phase 2
    if config.SAVE_TEST_SET:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        test_path = f"{config.OUTPUT_DIR}/test_set.csv"
        test_df.to_csv(test_path, index=False)
        print(f"\n✓ Saved test set: {test_path}")

    # Create prompts for training WITHOUT few-shot examples
    # NOTE: Few-shot examples cause the model to copy rather than generate
    print("\n✓ Creating Gemma-2-format narrative prompts...")
    train_prompts = []
    for idx, row in train_df.iterrows():
        prompt = create_narrative_prompt_with_examples(
            row['input'],
            row['target'],  # Using 'target' column for narrative
            example_pool=None  # NO EXAMPLES - direct learning
        )
        train_prompts.append(prompt)

    val_prompts = []
    for _, row in val_df.iterrows():
        prompt = create_narrative_prompt_with_examples(
            row['input'],
            row['target'],  # Using 'target' column for narrative
            example_pool=None  # NO EXAMPLES - direct learning
        )
        val_prompts.append(prompt)

    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict({"text": train_prompts})
    val_dataset = Dataset.from_dict({"text": val_prompts})

    print(f"✓ Created narrative datasets with few-shot examples:")
    print(f"  Train: {len(train_dataset)} prompts")
    print(f"  Val:   {len(val_dataset)} prompts")
    print(f"  ℹ️  Each prompt contains 2 real narrative examples from training data")

    return train_dataset, val_dataset, test_df, df


def load_model(config: GemmaNarrativeConfig):
    """Load Gemma-2-9B model with LoRA"""

    print("\n"+"="*80)
    print("MODEL LOADING")
    print("="*80)

    print(f"\n📥 Loading base model: {config.MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        load_in_4bit=config.LOAD_IN_4BIT,
    )

    print("✓ Base model loaded")

    print("\n🔧 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        target_modules=config.TARGET_MODULES,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.RANDOM_STATE,
    )

    print("✓ LoRA adapters added")
    print(f"  r={config.LORA_R}, alpha={config.LORA_ALPHA}")
    print(f"  Target modules: {config.TARGET_MODULES}")

    return model, tokenizer


def preprocess_logits_for_metrics(logits, labels):
    """
    MEMORY OPTIMIZATION: Convert logits to predictions before storing.
    This reduces memory from ~141GB (full logits) to ~300MB (token IDs only).
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics_wrapper(tokenizer):
    """Create a compute_metrics function with tokenizer in closure"""

    # Load metric calculators (lazy loading)
    if METRICS_AVAILABLE:
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")
        bertscore = evaluate.load("bertscore")

    def compute_metrics(eval_pred):
        """Compute ROUGE, METEOR, and BERTScore metrics during training/validation"""
        if not METRICS_AVAILABLE:
            return {"eval_loss": 0.0}

        predictions, labels = eval_pred

        # Convert to numpy array if not already
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Handle padding and convert to int
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id).astype(np.int64)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int64)

        # Decode to text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Extract only the response part (after assistant header)
        clean_preds = []
        clean_labels = []
        for pred, label in zip(decoded_preds, decoded_labels):
            # Find the response section for Gemma-2 format
            if "<|start_header_id|>assistant<|end_header_id|>" in pred:
                pred = pred.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            if "<|start_header_id|>assistant<|end_header_id|>" in label:
                label = label.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

            # Remove end tokens
            pred = pred.replace("<|eot_id|>", "").strip()
            label = label.replace("<|eot_id|>", "").strip()

            clean_preds.append(pred)
            clean_labels.append(label)

        decoded_preds = clean_preds
        decoded_labels = clean_labels

        # Filter empty predictions
        valid_pairs = [(p, l) for p, l in zip(decoded_preds, decoded_labels)
                       if p.strip() and l.strip()]

        if not valid_pairs:
            return {
                "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
                "meteor": 0.0,
                "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0
            }

        preds, labels = zip(*valid_pairs)

        try:
            # Compute ROUGE
            rouge_results = rouge.compute(predictions=preds, references=labels)

            # Compute METEOR
            meteor_result = meteor.compute(predictions=preds, references=labels)

            # Compute BERTScore (using smaller model for speed)
            bertscore_results = bertscore.compute(
                predictions=preds,
                references=labels,
                lang="en",
                model_type="distilbert-base-uncased"
            )

            return {
                "rouge1": rouge_results["rouge1"],
                "rouge2": rouge_results["rouge2"],
                "rougeL": rouge_results["rougeL"],
                "meteor": meteor_result["meteor"],
                "bertscore_precision": np.mean(bertscore_results["precision"]),
                "bertscore_recall": np.mean(bertscore_results["recall"]),
                "bertscore_f1": np.mean(bertscore_results["f1"])
            }
        except Exception as e:
            print(f"⚠️ Error computing metrics: {e}")
            return {
                "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
                "meteor": 0.0,
                "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0
            }

    return compute_metrics


class TrainingTimeCallback(TrainerCallback):
    """Callback to track total training time"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.total_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n⏱️  Training started at: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    def on_train_end(self, args, state, control, **kwargs):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        hours = int(self.total_time // 3600)
        minutes = int((self.total_time % 3600) // 60)
        seconds = int(self.total_time % 60)
        print(f"\n⏱️  Training ended at: {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total training time: {hours}h {minutes}m {seconds}s ({self.total_time:.2f}s)")


def train_model(model, tokenizer, train_dataset, val_dataset, config: GemmaNarrativeConfig):
    """Train Gemma-2 model for narrative generation"""

    print("\n"+"="*80)
    print("TRAINING - NARRATIVE GENERATION")
    print("="*80)

    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        warmup_steps=config.WARMUP_STEPS,
        max_grad_norm=config.MAX_GRAD_NORM,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type=config.LR_SCHEDULER,
        optim=config.OPTIMIZER,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        save_total_limit=2,
    )

    print(f"\n✓ Training configuration:")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Epochs: {config.NUM_EPOCHS} (FULL - early stopping DISABLED)")
    print(f"  Optimizer: {config.OPTIMIZER}")
    print(f"  Weight decay: {config.WEIGHT_DECAY}")
    print(f"  Output format: NARRATIVE (well-organized flowing paragraphs)")

    # Setup callbacks
    time_callback = TrainingTimeCallback()
    # DISABLED early stopping to ensure full 3 epochs of training
    # early_stopping_callback = EarlyStoppingCallback(
    #     early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
    #     early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD
    # )

    # Trainer with memory-optimized metrics
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        args=training_args,
        compute_metrics=compute_metrics_wrapper(tokenizer) if METRICS_AVAILABLE else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[time_callback],  # Removed early_stopping_callback to train full 3 epochs
    )

    print("\n🚀 Starting narrative summary training...")

    # Train
    train_result = trainer.train()

    print("\n✓ Training completed!")
    print(f"  Final train loss: {train_result.training_loss:.4f}")

    # Save training logs with training time
    save_training_logs(trainer, config, time_callback)

    return trainer, train_result, time_callback


def save_training_logs(trainer, config: GemmaNarrativeConfig, time_callback=None):
    """Save training and validation logs to files"""

    print("\n💾 Saving training logs...")

    log_history = trainer.state.log_history

    # Extract training losses
    train_data = []
    for entry in log_history:
        if 'loss' in entry:
            train_data.append({
                'step': entry['step'],
                'epoch': entry.get('epoch', 0),
                'loss': entry['loss'],
                'learning_rate': entry.get('learning_rate', 0)
            })

    # Extract validation losses and metrics
    val_data = []
    eval_metrics = []
    for entry in log_history:
        if 'eval_loss' in entry:
            val_data.append({
                'step': entry['step'],
                'epoch': entry.get('epoch', 0),
                'eval_loss': entry['eval_loss']
            })
            # Extract all evaluation metrics
            metrics_entry = {'step': entry['step'], 'epoch': entry.get('epoch', 0)}
            for key in entry:
                if key.startswith('eval_'):
                    metrics_entry[key] = entry[key]
            eval_metrics.append(metrics_entry)

    # Save as JSON
    train_log_path = f"{config.LOGS_DIR}/training_loss.json"
    with open(train_log_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  ✓ Training loss saved: {train_log_path}")

    val_log_path = f"{config.LOGS_DIR}/validation_loss.json"
    with open(val_log_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"  ✓ Validation loss saved: {val_log_path}")

    # Save evaluation metrics (ROUGE, METEOR, BERTScore)
    eval_metrics_path = f"{config.LOGS_DIR}/evaluation_metrics.json"
    with open(eval_metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"  ✓ Evaluation metrics saved: {eval_metrics_path}")

    # Save full log history
    full_log_path = f"{config.LOGS_DIR}/full_training_log.json"
    with open(full_log_path, 'w') as f:
        json.dump(log_history, f, indent=2)
    print(f"  ✓ Full log saved: {full_log_path}")

    # Save summary statistics
    summary = {
        'total_steps': train_data[-1]['step'] if train_data else 0,
        'final_epoch': train_data[-1]['epoch'] if train_data else 0,
        'final_train_loss': train_data[-1]['loss'] if train_data else 0,
        'final_val_loss': val_data[-1]['eval_loss'] if val_data else 0,
        'min_train_loss': min([d['loss'] for d in train_data]) if train_data else 0,
        'min_val_loss': min([d['eval_loss'] for d in val_data]) if val_data else 0,
        'num_train_logs': len(train_data),
        'num_val_logs': len(val_data),
    }

    # Add training time if available
    if time_callback and time_callback.total_time:
        summary['total_training_time_seconds'] = time_callback.total_time
        summary['total_training_time_formatted'] = f"{int(time_callback.total_time // 3600)}h {int((time_callback.total_time % 3600) // 60)}m {int(time_callback.total_time % 60)}s"

    summary_path = f"{config.LOGS_DIR}/training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary saved: {summary_path}")

    print(f"\n✓ All logs saved to: {config.LOGS_DIR}/")


def save_models(model, tokenizer, config: GemmaNarrativeConfig):
    """Save fine-tuned models"""

    print("\n"+"="*80)
    print("SAVING MODELS")
    print("="*80)

    # Save LoRA adapters
    print(f"\n💾 Saving LoRA adapters to: {config.FINAL_MODEL_DIR}")
    model.save_pretrained(config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(config.FINAL_MODEL_DIR)
    print("✓ LoRA adapters saved")

    # Save merged model (for easier deployment)
    print(f"\n💾 Attempting to save merged model to: {config.MERGED_MODEL_DIR}")
    try:
        model.save_pretrained_merged(
            config.MERGED_MODEL_DIR,
            tokenizer,
            save_method="merged_16bit"
        )
        print("✓ Merged model saved (16-bit)")
    except RuntimeError as e:
        print(f"⚠️  Merged model save failed (known Gemma-2 issue): {e}")
        print("✓ LoRA adapters saved successfully - use these for inference")

    print("\n✓ Model saving complete!")


def save_metadata(config: GemmaNarrativeConfig, train_result, df: pd.DataFrame):
    """Save metadata for Phase 2"""

    print("\n"+"="*80)
    print("SAVING METADATA FOR PHASE 2")
    print("="*80)

    # Calculate target statistics
    target_words = df['target'].str.split().str.len()

    metadata = {
        "model_info": {
            "base_model": config.MODEL_NAME,
            "fine_tuned_model": config.FINAL_MODEL_DIR,
            "merged_model": config.MERGED_MODEL_DIR,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "quantization": "4-bit",
            "output_format": "narrative (well-organized flowing paragraphs)"
        },
        "lora_config": {
            "r": config.LORA_R,
            "alpha": config.LORA_ALPHA,
            "dropout": config.LORA_DROPOUT,
            "target_modules": config.TARGET_MODULES,
        },
        "training_config": {
            "batch_size": config.BATCH_SIZE,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": config.LEARNING_RATE,
            "num_epochs": config.NUM_EPOCHS,
            "optimizer": config.OPTIMIZER,
        },
        "dataset_info": {
            "dataset_path": config.DATASET_PATH,
            "total_samples": len(df),
            "train_samples": 4500,
            "val_samples": 300,
            "test_samples": 200,
            "test_set_path": f"{config.OUTPUT_DIR}/test_set.csv",
            "target_column": "target (narrative summaries)",
            "avg_input_chars": float(df['input'].str.len().mean()),
            "avg_target_chars": float(df['target'].str.len().mean()),
            "avg_target_words": float(target_words.mean()),
            "target_words_min": int(target_words.min()),
            "target_words_max": int(target_words.max()),
            "target_words_median": float(target_words.median()),
        },
        "training_results": {
            "final_train_loss": float(train_result.training_loss),
            "total_steps": train_result.global_step,
        },
        "phase2_ready": {
            "rag_integration": True,
            "explainability": True,
            "test_set_available": True,
        },
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    metadata_path = f"{config.OUTPUT_DIR}/gemma_narrative_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved: {metadata_path}")
    print("\n✓ Phase 2 Preparation Complete:")
    print("  - Test set saved for narrative generation evaluation")
    print("  - Models ready for RAG integration")
    print("  - Configuration saved for explainability analysis")


def main():
    """Main training pipeline for narrative generation"""

    print("\n" + "="*80)
    print(" "*15 + "GEMMA-2-9B-INSTRUCT NARRATIVE FINE-TUNING")
    print(" "*8 + "Well-Organized Medical Narrative Summaries - 5000 Samples")
    print("="*80)

    # Initialize config
    config = GemmaNarrativeConfig()

    # Prepare dataset
    train_dataset, val_dataset, test_df, df = prepare_dataset(config)

    # Load model
    model, tokenizer = load_model(config)

    # Train
    trainer, train_result, time_callback = train_model(model, tokenizer, train_dataset, val_dataset, config)

    # Save models
    save_models(model, tokenizer, config)

    # Generate training plots
    create_narrative_training_plots(trainer, config, model_name="Gemma-2-9B Narrative", df=df)

    # Save metadata
    save_metadata(config, train_result, df)

    print("\n" + "="*80)
    print("NARRATIVE FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"\n📁 Output Directory: {config.OUTPUT_DIR}")
    print(f"   - LoRA Model: {config.FINAL_MODEL_DIR}")
    print(f"   - Merged Model: {config.MERGED_MODEL_DIR}")
    print(f"   - Test Set (200 samples): {config.OUTPUT_DIR}/test_set.csv")
    print(f"   - Plots: {config.PLOTS_DIR}")
    print(f"   - Metadata: {config.OUTPUT_DIR}/gemma_narrative_metadata.json")

    print("\n📊 Next Steps:")
    print("   1. Run narrative generation evaluation on 200 test samples")
    print("   2. Phase 2: RAG integration with narrative fine-tuned model")
    print("   3. Phase 2: Explainability analysis")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
