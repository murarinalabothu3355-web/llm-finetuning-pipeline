"""
prepare_data.py

Builds a small instruction-tuning dataset in JSONL format and formats
it into prompt/response pairs for causal LM fine-tuning.

This ships with a synthetic domain-specific dataset (data center /
infrastructure support Q&A) so the pipeline runs standalone. Swap in
your own JSONL of {"instruction": ..., "response": ...} pairs to
fine-tune on a different domain.

Usage:
    python src/prepare_data.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "instructions.jsonl"

# A small synthetic instruction-tuning dataset. In a real project this
# would be your domain-specific corpus (support tickets, docs Q&A, etc).
SEED_EXAMPLES = [
    {
        "instruction": "What does a UPS do in a data center?",
        "response": "A UPS (Uninterruptible Power Supply) provides short-term backup power to data center equipment during a power outage, bridging the gap until generators come online.",
    },
    {
        "instruction": "Explain the purpose of CRAC units.",
        "response": "CRAC (Computer Room Air Conditioning) units regulate temperature and humidity in a data center to keep server equipment within safe operating ranges.",
    },
    {
        "instruction": "What is LoRA in the context of LLM fine-tuning?",
        "response": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the original model weights and injects small trainable low-rank matrices into specific layers, drastically reducing the number of trainable parameters.",
    },
    {
        "instruction": "Why is class imbalance a problem in churn prediction?",
        "response": "Class imbalance means the model sees far more 'no churn' examples than 'churn' examples, so it can achieve high accuracy by just predicting the majority class while still missing most actual churners. Techniques like SMOTE rebalance the training data to address this.",
    },
    {
        "instruction": "What is the difference between precision and recall?",
        "response": "Precision measures how many of the predicted positive cases were actually positive, while recall measures how many of the actual positive cases the model correctly identified. There's typically a tradeoff between the two.",
    },
    {
        "instruction": "What does PEFT stand for and why is it useful?",
        "response": "PEFT stands for Parameter-Efficient Fine-Tuning. It's useful because it lets you adapt a large pretrained model to a new task by training only a small subset of parameters, saving memory and compute compared to full fine-tuning.",
    },
    {
        "instruction": "What is MLflow used for?",
        "response": "MLflow is an open-source platform for managing the ML lifecycle, including experiment tracking, model versioning, and deployment. It lets teams log metrics, parameters, and artifacts across training runs for reproducibility.",
    },
    {
        "instruction": "What is the role of a PDU in a data center?",
        "response": "A PDU (Power Distribution Unit) distributes electrical power from the main supply to multiple racks and devices within a data center, often with monitoring and remote control capabilities.",
    },
    {
        "instruction": "What is gradient accumulation and when would you use it?",
        "response": "Gradient accumulation sums gradients across multiple smaller batches before performing a weight update, allowing you to effectively train with a larger batch size than your GPU memory would normally allow.",
    },
    {
        "instruction": "Explain what SMOTE does.",
        "response": "SMOTE (Synthetic Minority Over-sampling Technique) generates new synthetic examples of the minority class by interpolating between existing minority-class data points and their nearest neighbors, helping balance an imbalanced dataset.",
    },
    {
        "instruction": "What is a checkpoint in model training?",
        "response": "A checkpoint is a saved snapshot of a model's weights (and often optimizer state) at a specific point during training, allowing you to resume training later or roll back to a known-good state.",
    },
    {
        "instruction": "What is the purpose of a learning rate warmup?",
        "response": "Learning rate warmup gradually increases the learning rate from a small value to the target value over the first few training steps, which helps stabilize training early on before the model's weights have adjusted.",
    },
    {
        "instruction": "What does an EPMS monitor in a data center?",
        "response": "An EPMS (Electrical Power Management System) monitors and manages electrical infrastructure such as switchgear, UPS units, and PDUs, providing real-time visibility into power load, faults, and capacity.",
    },
    {
        "instruction": "What is overfitting in machine learning?",
        "response": "Overfitting occurs when a model learns the training data too closely, including its noise, and as a result performs well on training data but poorly on new, unseen data.",
    },
    {
        "instruction": "Why do we use a validation set during training?",
        "response": "A validation set is held out from training and used to evaluate the model's performance during training, helping detect overfitting and guide decisions like early stopping or hyperparameter tuning.",
    },
    {
        "instruction": "What is the difference between SQL and NoSQL databases?",
        "response": "SQL databases store data in structured tables with fixed schemas and use SQL for queries, while NoSQL databases store data in flexible formats like documents or key-value pairs, often trading strict consistency for scalability.",
    },
    {
        "instruction": "What is root cause analysis in incident management?",
        "response": "Root cause analysis is the process of investigating an incident to identify the underlying reason it occurred, rather than just addressing the immediate symptoms, in order to prevent recurrence.",
    },
    {
        "instruction": "What does A/B testing mean in a model deployment context?",
        "response": "A/B testing in model deployment means routing a portion of live traffic to a new model version while the rest continues to the existing version, then comparing performance metrics to decide whether to fully roll out the new version.",
    },
    {
        "instruction": "What is the purpose of feature scaling in machine learning?",
        "response": "Feature scaling transforms numeric features to a common range or distribution, which helps algorithms that are sensitive to feature magnitude (like logistic regression or KNN) converge faster and perform more reliably.",
    },
    {
        "instruction": "What is the role of a wet cell battery in a data center UPS system?",
        "response": "Wet cell batteries store backup electrical energy and discharge it almost instantly when main power fails, providing power to critical systems until generators start and stabilize.",
    },
]


def build_dataset(n_repeats: int = 1, seed: int = 42) -> list[dict]:
    """In a real project this would load and clean a much larger corpus.
    Here we optionally repeat/shuffle the seed set to simulate a slightly
    larger dataset for demonstrating the train/eval split logic.
    """
    rng = random.Random(seed)
    examples = list(SEED_EXAMPLES) * n_repeats
    rng.shuffle(examples)
    return examples


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    examples = build_dataset()

    with open(OUTPUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} instruction/response pairs -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
