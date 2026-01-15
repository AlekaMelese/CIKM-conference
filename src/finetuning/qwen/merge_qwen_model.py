#!/usr/bin/env python3
"""
Standalone script to merge Qwen2-7B LoRA adapters with base model
Fixes cache directory issues and tensor size mismatch
"""

import os
import torch

# CRITICAL: Set cache directories to scratch BEFORE any imports
os.environ['HF_HOME'] = '~/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '~/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '~/.cache/huggingface/datasets'
os.environ['TORCH_HOME'] = './.cache/torch'

# Create cache directories
os.makedirs('~/.cache/huggingface', exist_ok=True)
os.makedirs('./.cache/torch', exist_ok=True)

print("="*80)
print("QWEN2-7B MODEL MERGE - SCRATCH DIRECTORY VERSION")
print("="*80)

print(f"\n✓ Cache directories set to scratch:")
print(f"  HF_HOME: {os.environ['HF_HOME']}")
print(f"  TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
print(f"  TORCH_HOME: {os.environ['TORCH_HOME']}")

# Now import unsloth after setting cache directories
try:
    from unsloth import FastLanguageModel
    print("\n✓ Unsloth imported successfully")
except ImportError as e:
    print(f"\n❌ Error importing unsloth: {e}")
    print("Please install: pip install unsloth")
    exit(1)

# Paths
LORA_MODEL_DIR = "./Final/Qwen/Finetuning/outputs/final_model"
MERGED_MODEL_DIR = "./Final/Qwen/Finetuning/outputs/merged_model"

print(f"\n📁 LoRA adapters: {LORA_MODEL_DIR}")
print(f"📁 Output directory: {MERGED_MODEL_DIR}")

# Load model with LoRA adapters
print("\n🔄 Loading LoRA adapters...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_MODEL_DIR,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    print("✓ LoRA adapters loaded successfully")
except Exception as e:
    print(f"❌ Error loading LoRA adapters: {e}")
    exit(1)

# Try different merge methods
print("\n" + "="*80)
print("ATTEMPTING MODEL MERGE")
print("="*80)

# Method 1: Try merged_16bit (original method)
print("\n📝 Method 1: Trying merged_16bit...")
try:
    model.save_pretrained_merged(
        MERGED_MODEL_DIR,
        tokenizer,
        save_method="merged_16bit"
    )
    print("✅ SUCCESS: Model merged with merged_16bit method")
    print(f"✓ Merged model saved to: {MERGED_MODEL_DIR}")
    exit(0)
except Exception as e:
    print(f"❌ Method 1 failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# Method 2: Try merged_4bit
print("\n📝 Method 2: Trying merged_4bit...")
try:
    model.save_pretrained_merged(
        f"{MERGED_MODEL_DIR}_4bit",
        tokenizer,
        save_method="merged_4bit"
    )
    print("✅ SUCCESS: Model merged with merged_4bit method")
    print(f"✓ Merged model saved to: {MERGED_MODEL_DIR}_4bit")
    exit(0)
except Exception as e:
    print(f"❌ Method 2 failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# Method 3: Try lora (keep adapters separate)
print("\n📝 Method 3: Trying lora (keep adapters separate)...")
try:
    model.save_pretrained_merged(
        f"{MERGED_MODEL_DIR}_lora",
        tokenizer,
        save_method="lora"
    )
    print("✅ SUCCESS: Model saved with LoRA adapters (not merged)")
    print(f"✓ Model saved to: {MERGED_MODEL_DIR}_lora")
    print("\nℹ️  Note: This keeps LoRA adapters separate but can be used for inference")
    exit(0)
except Exception as e:
    print(f"❌ Method 3 failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# Method 4: Manual merge using PEFT
print("\n📝 Method 4: Trying manual merge with PEFT...")
try:
    from peft import PeftModel

    # Load base model
    print("  Loading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2-7B-Instruct-bnb-4bit",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,  # Load in full precision for merging
    )

    # Load and merge adapters
    print("  Merging LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
    merged_model = model.merge_and_unload()

    # Save merged model
    print("  Saving merged model...")
    os.makedirs(f"{MERGED_MODEL_DIR}_peft", exist_ok=True)
    merged_model.save_pretrained(f"{MERGED_MODEL_DIR}_peft")
    tokenizer.save_pretrained(f"{MERGED_MODEL_DIR}_peft")

    print("✅ SUCCESS: Model merged manually with PEFT")
    print(f"✓ Merged model saved to: {MERGED_MODEL_DIR}_peft")
    exit(0)
except Exception as e:
    print(f"❌ Method 4 failed: {e}")
    print(f"   Error type: {type(e).__name__}")

print("\n" + "="*80)
print("❌ ALL MERGE METHODS FAILED")
print("="*80)
print("\n⚠️  The LoRA adapters are still available at:")
print(f"   {LORA_MODEL_DIR}")
print("\n💡 You can use the LoRA adapters directly for inference without merging.")
print("   Just load the model from the final_model directory.")
print("\n" + "="*80)
