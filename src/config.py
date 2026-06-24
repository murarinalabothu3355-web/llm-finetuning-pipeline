"""
config.py

Central configuration for the fine-tuning pipeline. Edit these values
to change the base model, LoRA hyperparameters, or training settings.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # Small, CPU-friendly default so this pipeline is runnable without a GPU.
    # Swap for a larger model (e.g. "meta-llama/Llama-3.2-1B" or "mistralai/Mistral-7B-v0.1")
    # once you have GPU access — the LoRA/PEFT code does not change.
    base_model_name: str = "distilgpt2"
    max_seq_length: int = 256


@dataclass
class LoRAConfig:
    r: int = 8                       # LoRA rank — higher = more capacity, more memory
    lora_alpha: int = 16             # scaling factor, typically 2x r
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["c_attn"])  # GPT-2 family attention proj
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42
    fp16: bool = False   # set True if running on a CUDA GPU
    report_to: str = "none"  # set to "mlflow" to enable MLflow tracking (see src/mlflow_tracking.py)


@dataclass
class DataConfig:
    dataset_path: str = "data/instructions.jsonl"
    train_split: float = 0.9
    prompt_template: str = (
        "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    )


MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoRAConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()
