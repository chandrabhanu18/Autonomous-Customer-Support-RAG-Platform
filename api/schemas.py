from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query_id: str
    response_id: str
    response_text: str
    intent: str
    intent_confidence: float
    retrieved_chunk_ids: list[str]
    faithfulness_score: Optional[float] = None
    relevance_score: Optional[float] = None


class EvaluateResponse(BaseModel):
    response_id: str
    faithfulness_score: float
    relevance_score: float
    combined_score: float


class FeedbackRequest(BaseModel):
    response_id: str
    rating: int
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    chunks_indexed: int
