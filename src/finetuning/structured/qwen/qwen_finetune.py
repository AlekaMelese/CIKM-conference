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

# Set cache directories to scratch (before other imports)
os.environ['HF_HOME'] = '~/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '~/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '~/.cache/huggingface/datasets'

# Import evaluation metrics
try:
    import evaluate
    METRICS_AVAILABLE = True
except ImportError:
    print("⚠️ evaluate library not available. Install: pip install evaluate")
    METRICS_AVAILABLE = False


class Qwen2Config:
    """Configuration for Qwen2-7B-Instruct fine-tuning on 5000 samples - A30 Optimized"""

    # Model settings (Qwen2-7B-Instruct with 128K context)
    MODEL_NAME = "unsloth/Qwen2-7B-Instruct-bnb-4bit"
    MAX_SEQ_LENGTH = 4096  # Can handle up to 128K, using 4K for medical summaries
    DTYPE = None  # Auto-detect
    LOAD_IN_4BIT = True

    # LoRA settings (optimized for Qwen2 architecture)
    LORA_R = 16  # Qwen2 works well with lower rank
    LORA_ALPHA = 32  # 2x LORA_R for Qwen2
    LORA_DROPOUT = 0.05  # ANTI-OVERFITTING: Regularization
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    # Training settings (A30 24GB optimized)
    BATCH_SIZE = 2  # A30-safe batch size
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size of 16
    LEARNING_RATE = 2e-4  # Qwen2 recommended learning rate
    NUM_EPOCHS = 3  # Standard for 5000 samples
    WARMUP_STEPS = 100  # Qwen2 benefits from warmup
    MAX_GRAD_NORM = 1.0
    WEIGHT_DECAY = 0.01  # ANTI-OVERFITTING: L2 regularization
    LR_SCHEDULER = "cosine"  # Cosine works better with Qwen2
    OPTIMIZER = "paged_adamw_8bit"  # Memory-efficient optimizer

    # ANTI-OVERFITTING: Early stopping configuration
    EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 evaluations
    EARLY_STOPPING_THRESHOLD = 0.001  # Minimum change to qualify as improvement

    # Dataset - 5000 samples
    DATASET_PATH = "./Data/5000_structured.csv"

    # Custom split: Test=200, Val=300, Train=4500
    NUM_TEST_SAMPLES = 200
    NUM_VAL_SAMPLES = 300
    # Train will be: total - test - val = 5000 - 200 - 300 = 4500
    RANDOM_STATE = 42

    # Output paths
    OUTPUT_DIR = "./Final/Qwen/Finetuning/outputs"
    CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
    FINAL_MODEL_DIR = f"{OUTPUT_DIR}/final_model"
    MERGED_MODEL_DIR = f"{OUTPUT_DIR}/merged_model"
    PLOTS_DIR = f"{OUTPUT_DIR}/plots"
    LOGS_DIR = f"{OUTPUT_DIR}/logs"

    # Logging
    LOGGING_STEPS = 10
    SAVE_STEPS = 100  # More frequent saves
    EVAL_STEPS = 50  # More frequent evaluation

    # Phase 2 preparation
    SAVE_METADATA = True
    SAVE_TEST_SET = True


def create_qwen2_prompt_with_examples(input_text: str, target_text: str = None,
                                       example_pool: list = None) -> str:
    """Create Qwen2-Instruct format prompt with 2 few-shot examples from actual training data"""

    # If no example pool provided, fall back to basic prompt
    if example_pool is None or len(example_pool) < 2:
        if target_text:
            prompt = f"""<|im_start|>system
You are a medical AI assistant specialized in generating structured discharge summaries.<|im_end|>
<|im_start|>user
Generate a structured discharge summary from the clinical note below.

Clinical Note:
{input_text}

Generate a structured summary with these sections:
- Case Type
- Patient & Service
- Chief Complaint / Admission Context
- History of Present Illness (HPI)
- Past Medical / Surgical History
- Medications (Discharge / Ongoing)
- Physical Examination
- Investigations / Labs / Imaging
- Assessment / Impression
- Discharge Condition
- Follow-Up & Recommendations<|im_end|>
<|im_start|>assistant
{target_text}<|im_end|>"""
        else:
            prompt = f"""<|im_start|>system
You are a medical AI assistant specialized in generating structured discharge summaries.<|im_end|>
<|im_start|>user
Generate a structured discharge summary from the clinical note below.

Clinical Note:
{input_text}

Generate a structured summary with these sections:
- Case Type
- Patient & Service
- Chief Complaint / Admission Context
- History of Present Illness (HPI)
- Past Medical / Surgical History
- Medications (Discharge / Ongoing)
- Physical Examination
- Investigations / Labs / Imaging
- Assessment / Impression
- Discharge Condition
- Follow-Up & Recommendations<|im_end|>
<|im_start|>assistant
"""
        return prompt

    # Randomly select 2 examples from the pool (not the current sample)
    import random
    examples = random.sample(example_pool, min(2, len(example_pool)))

    if target_text:
        # Training format with real few-shot examples
        prompt = f"""<|im_start|>system
You are a medical AI assistant specialized in generating structured discharge summaries.<|im_end|>
<|im_start|>user
Here are two examples of the correct format:

EXAMPLE 1:
Clinical Note: {examples[0]['input'][:1000]}...

Structured Summary: {examples[0]['structured_target'][:1500]}...

---

EXAMPLE 2:
Clinical Note: {examples[1]['input'][:1000]}...

Structured Summary: {examples[1]['structured_target'][:1500]}...

---

Now generate a structured summary for this new clinical note:

Clinical Note:
{input_text}

Generate a structured summary with these sections:
- Case Type
- Patient & Service
- Chief Complaint / Admission Context
- History of Present Illness (HPI)
- Past Medical / Surgical History
- Medications (Discharge / Ongoing)
- Physical Examination
- Investigations / Labs / Imaging
- Assessment / Impression
- Discharge Condition
- Follow-Up & Recommendations<|im_end|>
<|im_start|>assistant
{target_text}<|im_end|>"""
    else:
        # Inference format with real few-shot examples
        prompt = f"""<|im_start|>system
You are a medical AI assistant specialized in generating structured discharge summaries.<|im_end|>
<|im_start|>user
Here are two examples of the correct format:

EXAMPLE 1:
Clinical Note: {examples[0]['input'][:1000]}...

Structured Summary: {examples[0]['structured_target'][:1500]}...

---

EXAMPLE 2:
Clinical Note: {examples[1]['input'][:1000]}...

Structured Summary: {examples[1]['structured_target'][:1500]}...

---

Now generate a structured summary for this new clinical note:

Clinical Note:
{input_text}

Generate a structured summary with these sections:
- Case Type
- Patient & Service
- Chief Complaint / Admission Context
- History of Present Illness (HPI)
- Past Medical / Surgical History
- Medications (Discharge / Ongoing)
- Physical Examination
- Investigations / Labs / Imaging
- Assessment / Impression
- Discharge Condition
- Follow-Up & Recommendations<|im_end|>
<|im_start|>assistant
"""

    return prompt


def create_qwen2_prompt(input_text: str, target_text: str = None) -> str:
    """Original create prompt function - kept for backward compatibility"""
    return create_qwen2_prompt_with_examples(input_text, target_text, example_pool=None)


def prepare_dataset(config: Qwen2Config):
    """Load and split dataset with custom sizes: Train=4500, Val=300, Test=200"""

    print("="*80)
    print("DATASET PREPARATION - 5000 SAMPLES")
    print("="*80)

    # Load dataset
    df = pd.read_csv(config.DATASET_PATH)
    print(f"\n✓ Loaded dataset: {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")

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

    # Create example pool from training data for few-shot learning
    print("\n✓ Preparing few-shot example pool from training data...")
    example_pool = train_df[['input', 'structured_target']].to_dict('records')
    print(f"  ✓ Example pool size: {len(example_pool)} samples")
    print(f"  ℹ️  Each prompt will include 2 random examples from actual training data")

    # Create prompts for training with few-shot examples
    print("\n✓ Creating Qwen2-format prompts with few-shot examples...")
    train_prompts = []
    for idx, row in train_df.iterrows():
        # Create example pool excluding current sample to avoid data leakage
        other_examples = [ex for i, ex in enumerate(example_pool) if i != idx]
        prompt = create_qwen2_prompt_with_examples(
            row['input'],
            row['structured_target'],
            example_pool=other_examples
        )
        train_prompts.append(prompt)

    val_prompts = []
    for _, row in val_df.iterrows():
        # Use full training set as example pool for validation
        prompt = create_qwen2_prompt_with_examples(
            row['input'],
            row['structured_target'],
            example_pool=example_pool
        )
        val_prompts.append(prompt)

    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict({"text": train_prompts})
    val_dataset = Dataset.from_dict({"text": val_prompts})

    print(f"✓ Created datasets with few-shot examples:")
    print(f"  Train: {len(train_dataset)} prompts")
    print(f"  Val:   {len(val_dataset)} prompts")
    print(f"  ℹ️  Each prompt contains 2 real examples from training data")

    return train_dataset, val_dataset, test_df, df


def load_model(config: Qwen2Config):
    """Load Qwen2-7B-Instruct model with LoRA"""

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

    Without this, evaluation tries to store: (batch × seq_len × vocab_size) in float32
    With this, we only store: (batch × seq_len) in int64
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
        """Compute ROUGE, METEOR, and BERTScore metrics during training/validation

        NOTE: This uses teacher forcing (ground truth as context) which gives
        optimistic scores. For true generalization metrics, use generate() on
        the test set after training completes.
        """
        if not METRICS_AVAILABLE:
            return {"eval_loss": 0.0}

        predictions, labels = eval_pred

        # Convert to numpy array if not already
        predictions = np.array(predictions)
        labels = np.array(labels)

        # predictions are now token IDs (2D: batch × seq_len) thanks to preprocess_logits_for_metrics
        # No need to check for 3D shape or take argmax

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
            # Find the response section for Qwen2 format
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


def train_model(model, tokenizer, train_dataset, val_dataset, config: Qwen2Config):
    """Train Qwen2-7B-Instruct model"""

    print("\n"+"="*80)
    print("TRAINING")
    print("="*80)

    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,  # Match training batch size for eval
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
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        save_total_limit=2,
        # MEMORY OPTIMIZATION: Use preprocess_logits_for_metrics to convert logits to token IDs
        # This reduces memory from ~141GB to ~300MB during evaluation
    )

    print(f"\n✓ Training configuration:")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Optimizer: {config.OPTIMIZER}")
    print(f"  Weight decay: {config.WEIGHT_DECAY} (regularization)")
    print(f"  LoRA dropout: {config.LORA_DROPOUT} (regularization)")
    print(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"  Evaluation metrics: ROUGE, METEOR, BERTScore (memory-optimized)")
    print(f"  Memory optimization: Using preprocess_logits_for_metrics (~300MB vs ~141GB)")

    # Setup callbacks
    time_callback = TrainingTimeCallback()
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD
    )

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
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # MEMORY OPTIMIZATION
        callbacks=[time_callback, early_stopping_callback],
    )

    print("\n🚀 Starting training...")

    # Train
    train_result = trainer.train()

    print("\n✓ Training completed!")
    print(f"  Final train loss: {train_result.training_loss:.4f}")

    # Save training logs with training time
    save_training_logs(trainer, config, time_callback)

    return trainer, train_result, time_callback


def save_training_logs(trainer, config: Qwen2Config, time_callback=None):
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

    # Extract validation losses
    val_data = []
    for entry in log_history:
        if 'eval_loss' in entry:
            val_data.append({
                'step': entry['step'],
                'epoch': entry.get('epoch', 0),
                'eval_loss': entry['eval_loss']
            })

    # Save as JSON
    train_log_path = f"{config.LOGS_DIR}/training_loss.json"
    with open(train_log_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  ✓ Training loss saved: {train_log_path}")

    val_log_path = f"{config.LOGS_DIR}/validation_loss.json"
    with open(val_log_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"  ✓ Validation loss saved: {val_log_path}")

    # Save full log history
    full_log_path = f"{config.LOGS_DIR}/full_training_log.json"
    with open(full_log_path, 'w') as f:
        json.dump(log_history, f, indent=2)
    print(f"  ✓ Full log saved: {full_log_path}")

    # Save as CSV for easy analysis
    if train_data:
        import csv
        csv_path = f"{config.LOGS_DIR}/training_loss.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'epoch', 'loss', 'learning_rate'])
            writer.writeheader()
            writer.writerows(train_data)
        print(f"  ✓ Training CSV saved: {csv_path}")

    if val_data:
        csv_path = f"{config.LOGS_DIR}/validation_loss.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'epoch', 'eval_loss'])
            writer.writeheader()
            writer.writerows(val_data)
        print(f"  ✓ Validation CSV saved: {csv_path}")

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


def save_models(model, tokenizer, config: Qwen2Config):
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
    print(f"\n💾 Saving merged model to: {config.MERGED_MODEL_DIR}")
    model.save_pretrained_merged(
        config.MERGED_MODEL_DIR,
        tokenizer,
        save_method="merged_16bit"
    )
    print("✓ Merged model saved (16-bit)")

    print("\n✓ All models saved successfully!")


def create_publication_plots(trainer, train_result, config: Qwen2Config, df: pd.DataFrame, time_callback=None):
    """Generate comprehensive publication-quality plots with all metrics"""

    print("\n"+"="*80)
    print("GENERATING COMPREHENSIVE TRAINING VISUALIZATIONS")
    print("="*80)

    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    log_history = trainer.state.log_history

    # Extract all data
    train_losses = [x['loss'] for x in log_history if 'loss' in x]
    train_steps = [x['step'] for x in log_history if 'loss' in x]
    train_epochs = [x.get('epoch', 0) for x in log_history if 'loss' in x]

    eval_data = [x for x in log_history if 'eval_loss' in x]
    eval_losses = [x['eval_loss'] for x in eval_data]
    eval_steps = [x['step'] for x in eval_data]
    eval_epochs = [x.get('epoch', 0) for x in eval_data]

    # Extract ROUGE scores
    rouge1_scores = [x.get('eval_rouge1', 0) for x in eval_data]
    rouge2_scores = [x.get('eval_rouge2', 0) for x in eval_data]
    rougeL_scores = [x.get('eval_rougeL', 0) for x in eval_data]

    # Extract METEOR scores
    meteor_scores = [x.get('eval_meteor', 0) for x in eval_data]

    # Extract BERTScore
    bertscore_f1 = [x.get('eval_bertscore_f1', 0) for x in eval_data]
    bertscore_precision = [x.get('eval_bertscore_precision', 0) for x in eval_data]
    bertscore_recall = [x.get('eval_bertscore_recall', 0) for x in eval_data]

    # Extract learning rate and gradient norm
    learning_rates = [x.get('learning_rate', 0) for x in log_history if 'learning_rate' in x]
    lr_steps = [x['step'] for x in log_history if 'learning_rate' in x]
    grad_norms = [x.get('grad_norm', 0) for x in log_history if 'grad_norm' in x]
    grad_steps = [x['step'] for x in log_history if 'grad_norm' in x]

    # PLOT 1: Training & Validation Loss with Overfitting Detection
    print("\n  Creating Plot 1: Training & Validation Loss (Overfitting Detection)...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    ax.plot(train_steps, train_losses, label='Training Loss', color='#3498db', linewidth=2.5, alpha=0.8)
    ax.plot(eval_steps, eval_losses, label='Validation Loss', color='#e74c3c', linewidth=2.5, linestyle='--', marker='o', markersize=6)

    # Find best validation loss point
    if eval_losses:
        min_val_idx = eval_losses.index(min(eval_losses))
        ax.axvline(eval_steps[min_val_idx], color='green', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'Best Val Loss (step {eval_steps[min_val_idx]})')
        ax.scatter([eval_steps[min_val_idx]], [eval_losses[min_val_idx]], color='green', s=200, zorder=5, marker='*')

    # Overfitting zone annotation
    if len(eval_losses) > 1 and eval_losses[-1] > eval_losses[min_val_idx] * 1.1:
        ax.axvspan(eval_steps[min_val_idx], eval_steps[-1], alpha=0.15, color='red',
                   label='Potential Overfitting Zone')
        ax.text(0.5, 0.95, '⚠️ OVERFITTING DETECTED: Validation loss increasing while training loss decreasing',
                transform=ax.transAxes, fontsize=11, color='red', fontweight='bold',
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training & Validation Loss: Overfitting Analysis (5000 Samples)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/1_training_validation_loss.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.PLOTS_DIR}/1_training_validation_loss.pdf', bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 1_training_validation_loss.png/.pdf")

    # PLOT 2: ROUGE Scores Evolution
    print("\n  Creating Plot 2: ROUGE Scores Over Training...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    if rouge1_scores and any(rouge1_scores):
        ax.plot(eval_epochs, rouge1_scores, label='ROUGE-1', color='#e74c3c', linewidth=2.5, marker='o', markersize=7)
        ax.plot(eval_epochs, rouge2_scores, label='ROUGE-2', color='#3498db', linewidth=2.5, marker='s', markersize=7)
        ax.plot(eval_epochs, rougeL_scores, label='ROUGE-L', color='#2ecc71', linewidth=2.5, marker='^', markersize=7)

        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('ROUGE Score', fontsize=13, fontweight='bold')
        ax.set_title('ROUGE Scores Evolution During Training (5000 Samples)', fontsize=15, fontweight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/2_rouge_scores.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.PLOTS_DIR}/2_rouge_scores.pdf', bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 2_rouge_scores.png/.pdf")

    # PLOT 3: METEOR and BERTScore Evolution
    print("\n  Creating Plot 3: METEOR & BERTScore Over Training...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # METEOR
    if meteor_scores and any(meteor_scores):
        ax1.plot(eval_epochs, meteor_scores, label='METEOR', color='#9b59b6', linewidth=2.5, marker='D', markersize=7)
        ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax1.set_ylabel('METEOR Score', fontsize=13, fontweight='bold')
        ax1.set_title('METEOR Score Evolution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

    # BERTScore
    if bertscore_f1 and any(bertscore_f1):
        ax2.plot(eval_epochs, bertscore_precision, label='Precision', color='#e74c3c', linewidth=2.5, marker='o', markersize=6)
        ax2.plot(eval_epochs, bertscore_recall, label='Recall', color='#3498db', linewidth=2.5, marker='s', markersize=6)
        ax2.plot(eval_epochs, bertscore_f1, label='F1', color='#2ecc71', linewidth=3, marker='^', markersize=7)
        ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax2.set_ylabel('BERTScore', fontsize=13, fontweight='bold')
        ax2.set_title('BERTScore Evolution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/3_meteor_bertscore.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.PLOTS_DIR}/3_meteor_bertscore.pdf', bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 3_meteor_bertscore.png/.pdf")

    # PLOT 4: Learning Rate & Gradient Norm
    print("\n  Creating Plot 4: Learning Rate & Gradient Norm...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Learning rate
    if learning_rates:
        ax1.plot(lr_steps, learning_rates, color='#e67e22', linewidth=2.5)
        ax1.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
        ax1.set_title('Learning Rate Schedule (Linear Decay)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

    # Gradient norm
    if grad_norms:
        ax2.plot(grad_steps, grad_norms, color='#16a085', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=config.MAX_GRAD_NORM, color='red', linestyle='--', linewidth=2, label=f'Max Grad Norm ({config.MAX_GRAD_NORM})')
        ax2.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Gradient Norm', fontsize=13, fontweight='bold')
        ax2.set_title('Gradient Norm (Clipping Monitor)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/4_lr_gradient.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.PLOTS_DIR}/4_lr_gradient.pdf', bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 4_lr_gradient.png/.pdf")

    # PLOT 5: All Metrics Dashboard (6 subplots)
    print("\n  Creating Plot 5: Comprehensive Metrics Dashboard...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comprehensive Training Metrics Dashboard (5000 Samples) - Qwen2-7B-Instruct', fontsize=18, fontweight='bold')

    # Loss curves
    ax = axes[0, 0]
    ax.plot(train_epochs, train_losses, label='Train Loss', color='#3498db', linewidth=2)
    ax.plot(eval_epochs, eval_losses, label='Val Loss', color='#e74c3c', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ROUGE scores
    ax = axes[0, 1]
    if rouge1_scores and any(rouge1_scores):
        ax.plot(eval_epochs, rouge1_scores, label='ROUGE-1', linewidth=2, marker='o')
        ax.plot(eval_epochs, rouge2_scores, label='ROUGE-2', linewidth=2, marker='s')
        ax.plot(eval_epochs, rougeL_scores, label='ROUGE-L', linewidth=2, marker='^')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('ROUGE Score', fontsize=11, fontweight='bold')
        ax.set_title('ROUGE Scores', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    # METEOR
    ax = axes[0, 2]
    if meteor_scores and any(meteor_scores):
        ax.plot(eval_epochs, meteor_scores, color='#9b59b6', linewidth=2.5, marker='D')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('METEOR Score', fontsize=11, fontweight='bold')
        ax.set_title('METEOR Score', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    # BERTScore
    ax = axes[1, 0]
    if bertscore_f1 and any(bertscore_f1):
        ax.plot(eval_epochs, bertscore_f1, label='F1', color='#2ecc71', linewidth=2.5, marker='^')
        ax.plot(eval_epochs, bertscore_precision, label='Precision', color='#e74c3c', linewidth=2, marker='o')
        ax.plot(eval_epochs, bertscore_recall, label='Recall', color='#3498db', linewidth=2, marker='s')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('BERTScore', fontsize=11, fontweight='bold')
        ax.set_title('BERTScore Components', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    # Learning Rate
    ax = axes[1, 1]
    if learning_rates:
        ax.plot(lr_steps, learning_rates, color='#e67e22', linewidth=2.5)
        ax.set_xlabel('Step', fontsize=11, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
        ax.set_title('LR Schedule', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Gradient Norm
    ax = axes[1, 2]
    if grad_norms:
        ax.plot(grad_steps, grad_norms, color='#16a085', linewidth=1.5, alpha=0.7)
        ax.axhline(y=config.MAX_GRAD_NORM, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Step', fontsize=11, fontweight='bold')
        ax.set_ylabel('Gradient Norm', fontsize=11, fontweight='bold')
        ax.set_title('Gradient Clipping', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/5_metrics_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.PLOTS_DIR}/5_metrics_dashboard.pdf', bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 5_metrics_dashboard.png/.pdf")

    # Continue with remaining plots...
    print(f"\n✓ All comprehensive plots saved to: {config.PLOTS_DIR}/")


def save_metadata(config: Qwen2Config, train_result, df: pd.DataFrame):
    """Save metadata for Phase 2"""

    print("\n"+"="*80)
    print("SAVING METADATA FOR PHASE 2")
    print("="*80)

    metadata = {
        "model_info": {
            "base_model": config.MODEL_NAME,
            "fine_tuned_model": config.FINAL_MODEL_DIR,
            "merged_model": config.MERGED_MODEL_DIR,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "quantization": "4-bit",
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
            "avg_input_tokens": float(df['input_token_len'].mean()),
            "avg_target_tokens": float(df['target_token_len'].mean()),
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

    metadata_path = f"{config.OUTPUT_DIR}/qwen2_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved: {metadata_path}")
    print("\n✓ Phase 2 Preparation Complete:")
    print("  - Test set saved for constrained generation evaluation")
    print("  - Models ready for RAG integration")
    print("  - Configuration saved for explainability analysis")


def main():
    """Main training pipeline"""

    print("\n" + "="*80)
    print(" "*15 + "LLAMA-3.1-8B-INSTRUCT FINE-TUNING")
    print(" "*10 + "Medical Discharge Summary Generation - 5000 Samples")
    print("="*80)

    # Initialize config
    config = Qwen2Config()

    # Prepare dataset
    train_dataset, val_dataset, test_df, df = prepare_dataset(config)

    # Load model
    model, tokenizer = load_model(config)

    # Train
    trainer, train_result, time_callback = train_model(model, tokenizer, train_dataset, val_dataset, config)

    # Save models
    save_models(model, tokenizer, config)

    # Create plots
    create_publication_plots(trainer, train_result, config, df, time_callback)

    # Save metadata
    save_metadata(config, train_result, df)

    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"\n📁 Output Directory: {config.OUTPUT_DIR}")
    print(f"   - LoRA Model: {config.FINAL_MODEL_DIR}")
    print(f"   - Merged Model: {config.MERGED_MODEL_DIR}")
    print(f"   - Test Set (200 samples): {config.OUTPUT_DIR}/test_set.csv")
    print(f"   - Plots: {config.PLOTS_DIR}")
    print(f"   - Metadata: {config.OUTPUT_DIR}/qwen2_metadata.json")

    print("\n📊 Next Steps:")
    print("   1. Run constrained generation evaluation on 200 test samples")
    print("   2. Phase 2: RAG integration with fine-tuned model")
    print("   3. Phase 2: Explainability analysis")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
