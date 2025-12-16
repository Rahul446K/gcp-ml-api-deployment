# job_api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import job_manager
from mock_worker import run_mock_training

app = FastAPI(title="TinkerClone Job API")

# Initialize DB
job_manager.init_db()

class JobRequest(BaseModel):
    dataset_path: str
    model_name: str = "tiiuae/falcon-7b-instruct"
    metadata: dict = {}

@app.post("/jobs")
def submit_job(req: JobRequest, background_tasks: BackgroundTasks):
    job_id = job_manager.create_job(req.dataset_path, req.model_name, req.metadata)
    # schedule the mock worker in background
    background_tasks.add_task(run_mock_training, job_id, req.dataset_path, req.model_name, req.metadata)
    return {"job_id": job_id, "status": "submitted"}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    # return main fields
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "model_name": job["model_name"]
    }

@app.get("/jobs/{job_id}/results")
def job_results(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "logs": job["logs"],
        "adapter_path": job["adapter_path"]
    }
