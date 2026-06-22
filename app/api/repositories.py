import re
from fastapi import APIRouter, HTTPException, BackgroundTasks, status

from app.schemas.api import IndexRepositoryRequest, IndexRepositoryResponse, RepositorySummaryResponse
from app.services.ingestion_service import IngestionService
from app.db.qdrant_client import qdrant_service
from app.github.github_client import GitHubClient

router = APIRouter(prefix="/repositories", tags=["Repositories"])

# Regex strictly matches GitHub repository URLs like: https://github.com/owner/repo
GITHUB_URL_PATTERN = re.compile(r"^https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?$")

@router.post("/index", response_model=IndexRepositoryResponse, status_code=status.HTTP_202_ACCEPTED)
async def index_repository(request: IndexRepositoryRequest, background_tasks: BackgroundTasks):
    """
    Accepts a GitHub repository URL, validates it, and triggers a background 
    ingestion process. Returns 202 Accepted immediately so the client is not blocked.
    """
    match = GITHUB_URL_PATTERN.match(request.repo_url.strip())
    if not match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid GitHub URL. Must be in the format https://github.com/owner/repo"
        )
        
    owner, repo = match.groups()
    repository_id = f"{owner}/{repo}"
    
    # Launch background ingestion
    background_tasks.add_task(IngestionService.process_repository, owner, repo)
    
    return IndexRepositoryResponse(repository_id=repository_id)

@router.get("/{repository_id:path}/summary", response_model=RepositorySummaryResponse)
async def get_repository_summary(repository_id: str):
    """
    Returns chunk statistics from the vector database grouped by source type.
    Must strictly use Qdrant as the only data source.
    """
    if not repository_id or "/" not in repository_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid repository_id format. Must be owner/repo"
        )
        
    owner, repo = repository_id.split("/", 1)
    
    # Fetch chunk stats from Qdrant
    stats = await qdrant_service.get_repository_summary(repository_id)
    
    # Fetch repository metadata from GitHub
    try:
        async with GitHubClient() as client:
            repo_data = await client.get_repository(owner, repo)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to fetch metadata from GitHub: {str(e)}"
        )
    
    return RepositorySummaryResponse(
        repository_id=repository_id,
        name=repo_data.get("name", repo),
        owner=repo_data.get("owner", owner),
        description=repo_data.get("description") or "",
        stars=repo_data.get("stars", 0),
        forks=repo_data.get("forks", 0),
        readme_chunks=stats["readme_chunks"],
        documentation_chunks=stats["documentation_chunks"],
        issue_chunks=stats["issue_chunks"],
        pull_request_chunks=stats["pull_request_chunks"],
        total_chunks=stats["total_chunks"]
    )
