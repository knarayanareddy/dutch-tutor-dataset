#!/usr/bin/env python3
"""
QLoRA fine-tuning for Dutch Tutor models (Qwen3-0.6B / Qwen2.5-0.5B / Gemma 3 1B) on Modal A10G.

Designed specifically to prevent overfitting on 2,000 samples:
  - Strict response-only loss masking (loss computed only on <reply_nl>...</vocab>)
  - Low-capacity LoRA adapter (r=16, alpha=32, dropout=0.05 across all linear layers)
  - 2 epochs maximum with cosine LR decay and 8% warmup
  - Continuous validation loss tracking every 25 steps (90/10 split)
  - Automatic checkpoint recovery for the lowest eval_loss model
  - Optional full 16-bit weight merge ready for GGUF/Android APK export

Usage on Modal:
  modal run scripts/train_modal.py --model qwen --epochs 2
  modal run scripts/train_modal.py --model gemma --epochs 2
"""

import json
import os
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

# -------------------------------------------------------------------------
# Modal App & Container Definition
# -------------------------------------------------------------------------
if modal is not None:
    app = modal.App("dutch-tutor-qlora")

    # Persistent volume to store model weights, adapters, and merged models
    models_volume = modal.Volume.from_name("dutch-tutor-models", create_if_missing=True)

    # CUDA container image with PyTorch, Transformers, PEFT, TRL, BitsAndBytes
    train_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.4.0",
            "transformers>=4.48.0,<4.50.0",
            "peft>=0.14.0",
            "trl>=0.14.0,<0.16.0",
            "bitsandbytes>=0.45.0",
            "accelerate>=1.2.0",
            "datasets>=3.2.0",
            "sentencepiece>=0.2.0",
            "hf-transfer>=0.1.8",
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .add_local_dir("data", remote_path="/root/data")
    )
else:
    app = None
    train_image = None
    models_volume = None


# Supported model registries
MODEL_CONFIGS = {
    "qwen": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "response_template": "<|im_start|>assistant\n",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "qwen3": {
        "repo_id": "Qwen/Qwen3-0.6B",
        "response_template": "<|im_start|>assistant\n",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "gemma": {
        "repo_id": "google/gemma-3-1b-it",
        "response_template": "<start_of_turn>model\n",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "gemma3": {
        "repo_id": "google/gemma-3-1b-it",
        "response_template": "<start_of_turn>model\n",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
}


# -------------------------------------------------------------------------
# Training Logic
# -------------------------------------------------------------------------
def run_fine_tuning(
    model_name: str = "qwen",
    epochs: int = 2,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    merge_weights: bool = True,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

    model_key = model_name.lower()
    if model_key in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[model_key]
        base_model_id = cfg["repo_id"]
        response_template = cfg["response_template"]
        target_modules = cfg["target_modules"]
    else:
        # Custom repo ID passed
        base_model_id = model_name
        response_template = "<|im_start|>assistant\n" if "qwen" in model_name.lower() else "<start_of_turn>model\n"
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    clean_tag = base_model_id.replace("/", "_")
    output_dir = f"/models/adapters/{clean_tag}"
    merged_dir = f"/models/merged/{clean_tag}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    print("=" * 70)
    print(f"🚀 Starting QLoRA Fine-Tuning on Modal A10G (24GB VRAM)")
    print(f"   Base Model:        {base_model_id}")
    print(f"   Target Modules:    {target_modules}")
    print(f"   LoRA Config:       r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"   Epochs:            {epochs} (anti-overfitting limit)")
    print(f"   Effective Batch:   {per_device_batch_size * gradient_accumulation_steps}")
    print(f"   Learning Rate:     {learning_rate} with cosine schedule")
    print("=" * 70)

    # 1. Load Data
    train_path = Path("/root/data/train.jsonl")
    val_path = Path("/root/data/val.jsonl")

    # Fallback to local files if not running in container root
    if not train_path.exists():
        train_path = Path("data/train.jsonl")
        val_path = Path("data/val.jsonl")

    def load_jsonl(p: Path):
        rows = []
        with p.open(encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    rows.append(json.loads(l))
        return rows

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    print(f"Loaded {len(train_data)} training rows and {len(val_data)} validation rows.")

    # 2. Tokenizer & Chat Template Formatting
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def format_conversations(row_list):
        formatted_texts = []
        for r in row_list:
            text = tokenizer.apply_chat_template(
                r["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted_texts.append(text)
        return Dataset.from_dict({"text": formatted_texts})

    train_dataset = format_conversations(train_data)
    val_dataset = format_conversations(val_data)

    print("Formatted prompt sample (last 250 chars):")
    print(repr(train_dataset[0]["text"][-250:]))

    # 3. 4-bit Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Adapter Setup
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Data Collator for Response-Only Masking
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 6. SFT Trainer Setup
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        max_seq_length=512,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        peft_config=peft_config,
        args=sft_config,
    )

    print("Starting training...")
    train_result = trainer.train()
    print("Training completed successfully!")
    print("Metrics:", train_result.metrics)

    # Save best adapter
    print(f"Saving best LoRA adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 7. Post-Training Validation Test
    print("\n" + "=" * 50)
    print("🧪 Running Quick Validation Test Generation")
    print("=" * 50)
    test_user_query = "Ik heb gisteren twee fiets gekocht maar een is kapot gegaan."
    test_system = "Je bent een Nederlandse taalcoach. Level: A2. [DB_HINTS: fiets=de, winkel=de, repareren=verb]"
    test_messages = [
        {"role": "system", "content": test_system},
        {"role": "user", "content": test_user_query},
    ]
    test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
    gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("Prompt:", test_user_query)
    print("Model Output:\n", gen_text)

    # 8. Merge Weights (16-bit Full Model for GGUF/Android)
    if merge_weights:
        print("\n" + "=" * 50)
        print(f"📦 Merging LoRA weights into 16-bit base model -> {merged_dir}")
        print("=" * 50)
        del model
        torch.cuda.empty_cache()

        base_merge = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_merge = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        merged_model = peft_merge.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        print(f"✅ Full 16-bit model saved to {merged_dir}")

    return {
        "status": "success",
        "output_adapter_dir": output_dir,
        "merged_dir": merged_dir if merge_weights else None,
        "train_loss": train_result.metrics.get("train_loss"),
    }


# -------------------------------------------------------------------------
# Modal Function Entrypoint
# -------------------------------------------------------------------------
if app is not None:
    hf_secrets = []
    if os.environ.get("HF_TOKEN"):
        hf_secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

    @app.function(
        image=train_image,
        gpu="A10G",
        volumes={"/models": models_volume},
        secrets=hf_secrets,
        timeout=3600,
    )
    def train_modal_entrypoint(
        model_name: str = "qwen",
        epochs: int = 2,
        lr: float = 2e-4,
        merge: bool = True,
    ):
        result = run_fine_tuning(
            model_name=model_name,
            epochs=epochs,
            learning_rate=lr,
            merge_weights=merge,
        )
        models_volume.commit()
        return result

    @app.local_entrypoint()
    def main(
        model: str = "qwen",
        epochs: int = 2,
        lr: float = 2e-4,
        merge: bool = True,
    ):
        print(f"Submitting QLoRA fine-tuning run to Modal for model: {model}...")
        res = train_modal_entrypoint.remote(
            model_name=model,
            epochs=epochs,
            lr=lr,
            merge=merge,
        )
        print("Run finished. Result:", res)


if __name__ == "__main__":
    if modal is None:
        print("Modal is not installed. To run locally or inspect options, install modal:")
        print("  pip install modal")
        print("Then run:")
        print("  modal run scripts/train_modal.py --model qwen")
    else:
        import sys
        print("To launch the job on Modal A10G:")
        print("  modal run scripts/train_modal.py --model qwen")
        print("  modal run scripts/train_modal.py --model gemma")
