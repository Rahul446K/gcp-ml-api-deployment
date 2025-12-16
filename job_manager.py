# job_manager.py
import sqlite3
import uuid
import json
from datetime import datetime
from typing import Optional

DB_PATH = "jobs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        created_at TEXT,
        updated_at TEXT,
        status TEXT,
        dataset_path TEXT,
        model_name TEXT,
        adapter_path TEXT,
        logs TEXT,
        metadata TEXT
    )
    """)
    conn.commit()
    conn.close()

def _now():
    return datetime.utcnow().isoformat() + "Z"

def create_job(dataset_path: str, model_name: str, metadata: dict):
    job_id = str(uuid.uuid4())
    created_at = _now()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
      INSERT INTO jobs(job_id, created_at, updated_at, status, dataset_path, model_name, adapter_path, logs, metadata)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (job_id, created_at, created_at, "pending", dataset_path, model_name, "", "", json.dumps(metadata)))
    conn.commit()
    conn.close()
    return job_id

def update_job_status(job_id: str, status: str, logs_append: Optional[str]=None, adapter_path: Optional[str]=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT logs FROM jobs WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    existing_logs = row[0] if row and row[0] else ""
    new_logs = existing_logs + ("\n" + logs_append if logs_append else "")
    updated_at = _now()
    # Use COALESCE to keep existing adapter_path if None
    if adapter_path:
        c.execute("""
          UPDATE jobs SET status = ?, updated_at = ?, logs = ?, adapter_path = ?
          WHERE job_id = ?
        """, (status, updated_at, new_logs, adapter_path, job_id))
    else:
        c.execute("""
          UPDATE jobs SET status = ?, updated_at = ?, logs = ?
          WHERE job_id = ?
        """, (status, updated_at, new_logs, job_id))
    conn.commit()
    conn.close()

def get_job(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT job_id, created_at, updated_at, status, dataset_path, model_name, adapter_path, logs, metadata FROM jobs WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "job_id": row[0],
        "created_at": row[1],
        "updated_at": row[2],
        "status": row[3],
        "dataset_path": row[4],
        "model_name": row[5],
        "adapter_path": row[6],
        "logs": row[7] or "",
        "metadata": json.loads(row[8] or "{}")
    }
