from __future__ import annotations

from pydantic import BaseModel, Field


class RecallDecisionSchema(BaseModel):
    recall_needed: bool = Field(
        default=False,
        description="True if the student is asking to revisit or continue something taught earlier.",
    )
    recall_reason: str = Field(
        default="",
        description="Short reason why recall is or is not needed.",
    )
    likely_topic: str = Field(
        default="",
        description="Best guess of the earlier topic the student is referring to.",
    )
    wants_old_example: bool = Field(
        default=False,
        description="True if the student seems to want a previous example again.",
    )
    wants_old_explanation_style: bool = Field(
        default=False,
        description="True if the student seems to want the earlier teaching style again.",
    )

    # New fields for feature 1
    topic_clear_for_recall: bool = Field(
        default=False,
        description=(
            "True only when the referenced earlier topic is clear enough to safely retrieve memory. "
            "False when the student says vague things like 'again', 'that', or 'same as before' "
            "without enough topic detail."
        ),
    )
    needs_recall_clarification: bool = Field(
        default=False,
        description=(
            "True when the student appears to be referring to something earlier, but the topic is too vague "
            "to safely recall."
        ),
    )
    clarification_question: str = Field(
        default="",
        description=(
            "A short clarification question to ask if the earlier topic is unclear, for example asking "
            "which exact part the student wants again."
        ),
    )
    fresh_teach_topic: str = Field(
        default="",
        description=(
            "The topic that can be offered for fresh teaching if recall is unclear. "
            "Usually this is the current or most likely topic."
        ),
    )


class MemoryCardExtractionSchema(BaseModel):
    should_create_memory: bool = Field(
        default=True,
        description="Set to true for any teaching content. Only false for greetings, thanks, or filler.",
    )
    topic: str = Field(default="", description="Short topic label.")
    confusion: str = Field(default="", description="Student confusion if any.")
    helpful_example: str = Field(default="", description="Example or analogy used.")
    student_preference: str = Field(default="", description="Student preference if any.")
    status: str = Field(default="", description="Short learning status.")
    snippet: str = Field(default="", description="Short summary of what was taught.")
    retrieval_text: str = Field(default="", description="Key facts and answers for recall.")