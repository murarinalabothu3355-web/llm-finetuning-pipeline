# Dockerfile for serving the fine-tuned LoRA adapter via a small FastAPI app.
#
# Build:
#   docker build -t llm-finetuning-pipeline .
# Run:
#   docker run -p 8000:8000 llm-finetuning-pipeline

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY outputs/final_adapter ./outputs/final_adapter

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
