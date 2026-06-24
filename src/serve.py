"""
serve.py

Minimal FastAPI app that serves the fine-tuned LoRA adapter for
inference over HTTP. Loads the model once at startup, then serves
requests from memory.

>>> Same caveat as train.py / inference.py — written against stable,
documented FastAPI + transformers/peft APIs, not executed in this
sandbox (no GPU/internet/libraries available there). <<<

Run locally:
    uvicorn src.serve:app --reload

Run in Docker:
    docker build -t llm-finetuning-pipeline .
    docker run -p 8000:8000 llm-finetuning-pipeline

Then:
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"instruction": "What does a UPS do in a data center?"}'
"""

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from config import MODEL_CONFIG
from inference import load_model, generate, validate_output

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = ROOT / "outputs" / "final_adapter"

app = FastAPI(title="LLM Fine-Tuning Inference API", version="1.0")

_model = None
_tokenizer = None


class GenerateRequest(BaseModel):
    instruction: str
    max_new_tokens: int = 120


class GenerateResponse(BaseModel):
    instruction: str
    response: str
    valid: bool
    validation_note: str = ""


@app.on_event("startup")
def load_model_on_startup():
    global _model, _tokenizer
    if not ADAPTER_PATH.exists():
        print(f"[warning] adapter not found at {ADAPTER_PATH}. /generate will fail until you train and place it there.")
        return
    _model, _tokenizer = load_model(str(ADAPTER_PATH))
    print("Model loaded and ready.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if _model is None:
        return GenerateResponse(
            instruction=req.instruction,
            response="",
            valid=False,
            validation_note="Model not loaded — adapter missing. Run src/train.py first.",
        )

    response_text = generate(_model, _tokenizer, req.instruction, req.max_new_tokens)
    is_valid, note = validate_output(response_text)

    return GenerateResponse(
        instruction=req.instruction,
        response=response_text,
        valid=is_valid,
        validation_note=note,
    )
