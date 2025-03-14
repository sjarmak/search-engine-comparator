from pydantic import BaseModel
from typing import List, Dict, Optional

class SearchRequest(BaseModel):
    query: str
    sources: List[str]
    metrics: List[str]
    fields: List[str]

class SearchResult(BaseModel):
    title: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    abstract: Optional[str]
    source: str
    rank: int
    url: Optional[str] 