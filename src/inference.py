"""
inference.py

Loads a base model + LoRA adapter and generates responses to new
instructions. Includes basic output validation (length and emptiness
checks) before returning a result, as a simple stand-in for the
output-validation step you'd want in a real serving pipeline.

>>> Same caveat as train.py: requires torch/transformers/peft installed
and was written against their stable APIs but not executed in this
sandbox (no GPU/internet/libraries available there). <<<

Usage:
    python src/inference.py --prompt "What does a UPS do in a data center?"
    python src/inference.py --adapter-path outputs/final_adapter
"""

import argparse
from pathlib import Path

from config import MODEL_CONFIG, DATA_CONFIG

ROOT = Path(__file__).resolve().parent.parent


def validate_output(text: str, min_length: int = 5) -> tuple[bool, str]:
    """Minimal output validation: rejects empty or suspiciously short
    generations. A real production system would add more checks here
    (toxicity filters, schema validation for structured outputs, etc).
    """
    cleaned = text.strip()
    if len(cleaned) < min_length:
        return False, "Output too short — likely a generation failure."
    if cleaned.lower().startswith("### instruction"):
        return False, "Model echoed the prompt template instead of generating a response."
    return True, ""


def load_model(adapter_path: str, base_model_override: str = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = base_model_override or MODEL_CONFIG.base_model_name

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, instruction: str, max_new_tokens: int = 120) -> str:
    import torch

    prompt = DATA_CONFIG.prompt_template.format(instruction=instruction, response="")
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # strip the prompt portion off, return only the generated continuation
    response = full_text[len(prompt):].strip()
    return response


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned LoRA adapter")
    parser.add_argument("--prompt", type=str, required=True, help="Instruction to ask the model")
    parser.add_argument("--adapter-path", type=str, default="outputs/final_adapter")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    adapter_path = ROOT / args.adapter_path
    if not adapter_path.exists():
        print(f"[error] adapter not found at {adapter_path}. Run src/train.py first.")
        return

    model, tokenizer = load_model(str(adapter_path), args.base_model)
    response = generate(model, tokenizer, args.prompt, args.max_new_tokens)

    is_valid, reason = validate_output(response)

    print(f"Instruction: {args.prompt}\n")
    print(f"Response: {response}\n")
    if not is_valid:
        print(f"[validation warning] {reason}")
    else:
        print("[validation passed]")


if __name__ == "__main__":
    main()
