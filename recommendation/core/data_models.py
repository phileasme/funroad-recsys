from pydantic import BaseModel, Field
from typing import List, Optional

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=10, ge=1, le=100)
    num_candidates: int = Field(default=100, ge=10, le=500)
    id: Optional[str] = Field(default="", min_length=1, max_length=1000)

class SearchResult(BaseModel):
    score: float
    name: str
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    ratings_count: Optional[int] = None
    ratings_score: Optional[float] = None
    price_cents: Optional[int] = None
    url: Optional[str] = None
    id: Optional[str] = None
    seller_id: Optional[str] = None
    seller_name: Optional[str] = None
    seller_thumbnail: Optional[str] = None
    score_origin: Optional[str] = None
    base_score: Optional[float]= None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time_ms: float

class HealthStatus(BaseModel):
    status: str
    elasticsearch: bool
    is_model_loaded: bool
    device: str