from fastapi import FastAPI
from .api.research import research

app = FastAPI()
app.include_router(research)
