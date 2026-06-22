from pydantic import BaseModel
from typing import List

class IndexRepositoryRequest(BaseModel):
    repo_url: str

class IndexRepositoryResponse(BaseModel):
    repository_id: str

class RepositorySummaryResponse(BaseModel):
    repository_id: str
    name: str
    owner: str
    description: str
    stars: int
    forks: int
    readme_chunks: int
    documentation_chunks: int
    issue_chunks: int
    pull_request_chunks: int
    total_chunks: int

class ChatRequest(BaseModel):
    repository_id: str
    question: str

class Citation(BaseModel):
    source_type: str
    title: str
    url: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    repository_id: str

class TestQuery(BaseModel):
    query: str
    expected_keywords: List[str]
    expected_source_types: List[str]

class BenchmarkRequest(BaseModel):
    repository_id: str
    test_queries: List[TestQuery]

class BenchmarkResult(BaseModel):
    query: str
    dense_precision_at_k: float
    bm25_precision_at_k: float
    hybrid_precision_at_k: float
    improvement: str
    error: str = None

class BenchmarkOverallMetrics(BaseModel):
    avg_dense_precision_at_5: float
    avg_bm25_precision_at_5: float
    avg_hybrid_precision_at_5: float
    best_strategy: str

class BenchmarkResponse(BaseModel):
    results: List[BenchmarkResult]
    overall_metrics: BenchmarkOverallMetrics
