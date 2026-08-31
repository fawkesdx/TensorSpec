import asyncio
import uuid
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

# In-memory store for job statuses
jobs: Dict[str, Dict[str, Any]] = {}

class JobPayload(BaseModel):
    # Dummy job payload
    name: str = "default_job"
    duration_seconds: int = 10

async def dummy_training_loop(job_id: str, duration: int):
    jobs[job_id]["status"] = "running"
    for i in range(duration):
        await asyncio.sleep(1)
        jobs[job_id]["progress"] = int(((i + 1) / duration) * 100)
    
    # Dump a placeholder results.npy to disk when done
    np.save("results.npy", np.array([1.0, 2.0, 3.0]))
    
    jobs[job_id]["status"] = "done"
    jobs[job_id]["progress"] = 100

@app.get("/ping")
async def ping():
    return {"status": "pong"}

@app.post("/jobs")
async def create_job(payload: JobPayload, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0
    }
    background_tasks.add_task(dummy_training_loop, job_id, payload.duration_seconds)
    return {"job_id": job_id}

@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9765)
