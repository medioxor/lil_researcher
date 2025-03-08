from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from ..agents.research import ResearchAgent
import uuid
import asyncio
from typing import Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel


# Job status enum
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Job model
class ResearchJob(BaseModel):
    id: str
    status: JobStatus
    question: str
    result: Optional[Any] = None
    error: Optional[str] = None


# Job queue
class ResearchJobQueue:
    def __init__(self):
        self.jobs: Dict[str, ResearchJob] = {}

    def add_job(self, question: str) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = ResearchJob(
            id=job_id, status=JobStatus.PENDING, question=question
        )
        return job_id

    def get_job(self, job_id: str) -> Optional[ResearchJob]:
        return self.jobs.get(job_id)


# Initialize router and agents
research = APIRouter(prefix="/v1/research")
research_agent = ResearchAgent()
job_queue = ResearchJobQueue()


# Background task to process jobs
async def process_research_job(job_id: str):
    job = job_queue.get_job(job_id)
    if not job:
        return

    job.status = JobStatus.RUNNING
    try:
        answer = await research_agent.research(job.question)
        job.result = answer
        job.status = JobStatus.COMPLETED
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)


# Submit a new research job
@research.post("/jobs")
async def create_research_job(question: str):
    job_id = job_queue.add_job(question)
    # Start processing in background
    asyncio.create_task(process_research_job(job_id))
    return {"job_id": job_id, "status": JobStatus.PENDING}


# Get job status and result
@research.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {"job_id": job.id, "status": job.status, "question": job.question}

    if job.status == JobStatus.COMPLETED:
        response["result"] = jsonable_encoder(job.result)
    elif job.status == JobStatus.FAILED:
        response["error"] = job.error

    return response


# Get all jobs
@research.get("/jobs")
async def get_all_jobs():
    return [
        {"job_id": job.id, "status": job.status, "question": job.question}
        for job in job_queue.jobs.values()
    ]
