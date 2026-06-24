"""
mlflow_tracking.py

Thin wrapper around MLflow for experiment tracking. Kept separate from
train.py so MLflow stays an optional dependency — only imported when
TrainingConfig.report_to == "mlflow".

Usage (standalone test of the wrapper logic):
    python src/mlflow_tracking.py --check
"""

import argparse


def start_run(base_model_name: str):
    import mlflow

    mlflow.set_experiment("llm-lora-finetuning")
    mlflow.start_run(run_name=f"lora-{base_model_name}")


def log_config(model_config, lora_config, training_config):
    import mlflow

    mlflow.log_params(
        {
            "base_model": model_config.base_model_name,
            "max_seq_length": model_config.max_seq_length,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": ",".join(lora_config.target_modules),
            "epochs": training_config.num_train_epochs,
            "batch_size": training_config.per_device_train_batch_size,
            "learning_rate": training_config.learning_rate,
        }
    )


def log_metrics(metrics: dict, step: int = None):
    import mlflow

    mlflow.log_metrics(metrics, step=step)


def log_adapter_artifact(adapter_dir: str):
    import mlflow

    mlflow.log_artifacts(adapter_dir, artifact_path="lora_adapter")


def end_run():
    import mlflow

    mlflow.end_run()


def _check_importable():
    """Quick standalone check: confirms this module's pure-Python logic
    (no mlflow import needed) is structurally sound. Run with --check.
    """
    import inspect

    functions = [start_run, log_config, log_metrics, log_adapter_artifact, end_run]
    for fn in functions:
        sig = inspect.signature(fn)
        print(f"{fn.__name__}{sig}")
    print("\n[ok] mlflow_tracking.py functions are defined and importable.")
    print("Actual MLflow calls require `pip install mlflow` and are not exercised by this check.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        _check_importable()
    else:
        print("Run with --check to verify this module's structure without requiring mlflow installed.")
