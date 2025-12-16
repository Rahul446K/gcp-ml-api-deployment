# mock_worker.py
import time
import job_manager
import os
import random
import json

ARTIFACTS_DIR = "job_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def run_mock_training(job_id: str, dataset_path: str, model_name: str, metadata: dict):
    try:
        job_manager.update_job_status(job_id, "running", logs_append=f"Job {job_id} started. model={model_name}, dataset={dataset_path}")
        total_steps = int(metadata.get("steps", 20))
        for step in range(1, total_steps + 1):
            loss = 5.0 / (step + 1) + random.random() * 0.05
            log_line = f"step={step}/{total_steps} loss={loss:.4f}"
            job_manager.update_job_status(job_id, "running", logs_append=log_line)
            time.sleep(0.4)  # small sleep to simulate time
        # Simulate saving adapter/checkpoint
        adapter_name = f"{job_id}_adapter"
        adapter_dir = os.path.join(ARTIFACTS_DIR, adapter_name)
        os.makedirs(adapter_dir, exist_ok=True)
        # Create a tiny metadata file to represent an adapter artifact
        with open(os.path.join(adapter_dir, "adapter_info.json"), "w") as f:
            json.dump({"job_id": job_id, "model": model_name, "note": "mock adapter"}, f)
        job_manager.update_job_status(job_id, "completed", logs_append="Training finished. Adapter saved.", adapter_path=adapter_dir)
    except Exception as e:
        job_manager.update_job_status(job_id, "failed", logs_append=f"Error: {str(e)}")
