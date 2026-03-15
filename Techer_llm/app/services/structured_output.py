from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RecallDecisionSchema(BaseModel):
    recall_needed: bool = Field(default=False)
    recall_reason: str = Field(default="")
    likely_topic: str = Field(default="")
    wants_old_example: bool = Field(default=False)
    wants_old_explanation_style: bool = Field(default=False)


class MemoryCardExtractionSchema(BaseModel):
    should_create_memory: bool = Field(default=False)
    topic: str = Field(default="")
    confusion: str = Field(default="")
    helpful_example: str = Field(default="")
    student_preference: str = Field(default="")
    status: str = Field(default="")
    snippet: str = Field(default="")
    retrieval_text: str = Field(default="")