from pydantic import BaseModel, Field
from typing import List, Optional

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=10, ge=1, le=100)
    num_candidates: int = Field(default=100, ge=10, le=500)

class SearchResult(BaseModel):
    score: float
    name: str
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time_ms: float

class HealthStatus(BaseModel):
    status: str
    elasticsearch: bool
    is_model_loaded: bool  # Renamed to avoid warning
    device: str