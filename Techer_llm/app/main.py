import logging

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

uvicorn_logger = logging.getLogger("uvicorn")


class LogRecord(BaseModel):
    level: str
    name: str
    message: str


@app.post("/internal/log")
def receive_log(record: LogRecord) -> dict:
    level = getattr(logging, record.level.upper(), logging.INFO)
    uvicorn_logger.log(level, "[terminal] %s | %s", record.name, record.message)
    return {"ok": True}


@app.get("/")
def root() -> dict:
    return {"message": "AI Teacher API is running"}