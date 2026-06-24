## Quick start

```bash
git clone https://github.com/<your-username>/llm-finetuning-pipeline.git
cd llm-finetuning-pipeline
pip install -r requirements.txt

python src/prepare_data.py              # generates data/instructions.jsonl
python src/train.py --dry-run           # validates config and data pipeline
python src/train.py                     # full LoRA fine-tuning run
python src/inference.py --prompt "What does a UPS do in a data center?"
```

## Serving with Docker

```bash
docker build -t llm-finetuning-pipeline .
docker run -p 8000:8000 llm-finetuning-pipeline

curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"instruction": "What is LoRA?"}'
```

## Enabling MLflow tracking

Set `report_to: str = "mlflow"` in `TrainingConfig` (`src/config.py`), install `mlflow`, then run:

```bash
mlflow ui   # in a separate terminal, view at http://localhost:5000
python src/train.py
```

## Tech stack

PyTorch, Hugging Face Transformers, PEFT (LoRA), Datasets, MLflow, FastAPI, Docker.

## License

MIT — use freely.