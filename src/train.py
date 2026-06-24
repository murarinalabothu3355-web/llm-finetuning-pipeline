"""
train.py

Fine-tunes a causal LM using LoRA (via the PEFT library) on an
instruction-following dataset.

>>> IMPORTANT — please read <<<
This script requires `torch`, `transformers`, `peft`, and `datasets`
to be installed, and needs internet access on first run to download
the base model from the Hugging Face Hub. It has been written against
the stable, documented APIs of `transformers` and `peft`, but — unlike
the rest of this repository — it has NOT been executed end-to-end in
the environment that built this repo, since that sandbox has no GPU,
no internet access, and none of these libraries installed.

Before relying on it, run:
    pip install -r requirements.txt
    python src/train.py --dry-run     # validates config + data loading only
    python src/train.py               # full training run

If you hit an API mismatch (library versions evolve), the most likely
spots are the `LoraConfig` field names and the `Trainer` arguments —
both are called out with comments below.

Usage:
    python src/train.py
    python src/train.py --dry-run
    python src/train.py --base-model gpt2 --epochs 1
"""

import argparse
import json
from pathlib import Path

from config import MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG, DATA_CONFIG

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_example(example: dict, template: str) -> str:
    return template.format(instruction=example["instruction"], response=example["response"])


def build_train_eval_split(examples: list[dict], train_split: float, seed: int):
    import random

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_split)
    return shuffled[:n_train], shuffled[n_train:]


def run_dry_run():
    """Validates config + data loading without touching any ML libraries.
    Useful to sanity-check the pipeline before installing heavy deps.
    """
    data_path = ROOT / DATA_CONFIG.dataset_path
    if not data_path.exists():
        print(f"[error] dataset not found at {data_path}. Run `python src/prepare_data.py` first.")
        return

    examples = load_jsonl(data_path)
    train_ex, eval_ex = build_train_eval_split(examples, DATA_CONFIG.train_split, TRAINING_CONFIG.seed)

    print(f"Loaded {len(examples)} examples ({len(train_ex)} train / {len(eval_ex)} eval)")
    print("\nSample formatted prompt:")
    print("-" * 60)
    print(format_example(train_ex[0], DATA_CONFIG.prompt_template))
    print("-" * 60)
    print(f"\nBase model: {MODEL_CONFIG.base_model_name}")
    print(f"LoRA config: r={LORA_CONFIG.r}, alpha={LORA_CONFIG.lora_alpha}, "
          f"target_modules={LORA_CONFIG.target_modules}")
    print(f"Training config: {TRAINING_CONFIG.num_train_epochs} epochs, "
          f"batch size {TRAINING_CONFIG.per_device_train_batch_size}, "
          f"lr {TRAINING_CONFIG.learning_rate}")
    print("\n[dry-run complete] Config and data pipeline look correct.")
    print("Run without --dry-run (and with torch/transformers/peft installed) to train for real.")


def run_training(base_model_override: str = None, epochs_override: int = None):
    # Imports are deliberately deferred to inside this function so that
    # --dry-run works even if these heavy libraries aren't installed yet.
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType

    base_model_name = base_model_override or MODEL_CONFIG.base_model_name
    num_epochs = epochs_override or TRAINING_CONFIG.num_train_epochs

    print(f"Loading base model and tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # ---- LoRA setup ----------------------------------------------------
    lora_config = LoraConfig(
        r=LORA_CONFIG.r,
        lora_alpha=LORA_CONFIG.lora_alpha,
        lora_dropout=LORA_CONFIG.lora_dropout,
        target_modules=LORA_CONFIG.target_modules,
        bias=LORA_CONFIG.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # confirms LoRA drastically cuts trainable params

    # ---- Data ------------------------------------------------------------
    data_path = ROOT / DATA_CONFIG.dataset_path
    examples = load_jsonl(data_path)
    train_ex, eval_ex = build_train_eval_split(examples, DATA_CONFIG.train_split, TRAINING_CONFIG.seed)

    def to_text(batch):
        texts = [
            format_example({"instruction": i, "response": r}, DATA_CONFIG.prompt_template)
            for i, r in zip(batch["instruction"], batch["response"])
        ]
        return tokenizer(texts, truncation=True, max_length=MODEL_CONFIG.max_seq_length, padding="max_length")

    train_dataset = Dataset.from_list(train_ex).map(to_text, batched=True)
    eval_dataset = Dataset.from_list(eval_ex).map(to_text, batched=True) if eval_ex else None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- Optional MLflow tracking ----------------------------------------
    if TRAINING_CONFIG.report_to == "mlflow":
        from mlflow_tracking import start_run, log_config

        start_run(base_model_name)
        log_config(MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG)

    training_args = TrainingArguments(
        output_dir=str(ROOT / TRAINING_CONFIG.output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=TRAINING_CONFIG.per_device_train_batch_size,
        per_device_eval_batch_size=TRAINING_CONFIG.per_device_eval_batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG.gradient_accumulation_steps,
        learning_rate=TRAINING_CONFIG.learning_rate,
        weight_decay=TRAINING_CONFIG.weight_decay,
        warmup_ratio=TRAINING_CONFIG.warmup_ratio,
        logging_steps=TRAINING_CONFIG.logging_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=TRAINING_CONFIG.eval_steps,
        save_steps=TRAINING_CONFIG.save_steps,
        save_total_limit=TRAINING_CONFIG.save_total_limit,
        seed=TRAINING_CONFIG.seed,
        fp16=TRAINING_CONFIG.fp16,
        report_to=[] if TRAINING_CONFIG.report_to in ("none", "mlflow") else [TRAINING_CONFIG.report_to],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    final_dir = ROOT / "outputs" / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved LoRA adapter -> {final_dir}")

    if TRAINING_CONFIG.report_to == "mlflow":
        from mlflow_tracking import end_run

        end_run()


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data without ML libraries")
    parser.add_argument("--base-model", type=str, default=None, help="Override base model name")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
    else:
        run_training(base_model_override=args.base_model, epochs_override=args.epochs)


if __name__ == "__main__":
    main()
