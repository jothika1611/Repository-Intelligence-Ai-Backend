import base64
import logging
import re
from typing import Optional, List, Dict, Any, Type

import httpx
from cachetools import TTLCache

from app.schemas.config import settings

logger = logging.getLogger(__name__)

# Module-level cache to persist across client instantiations
_github_cache = TTLCache(maxsize=100, ttl=900)

class GitHubClientError(Exception):
    """Base exception for GitHub client errors."""
    pass


class ResourceNotFoundError(GitHubClientError):
    """Raised when a specific GitHub resource (file, branch) cannot be found (404)."""
    pass


class RepositoryNotFoundError(GitHubClientError):
    """Raised when a repository cannot be found (404)."""
    pass


class GitHubRateLimitError(GitHubClientError):
    """Raised when GitHub API rate limit is exceeded (403 or headers)."""
    pass


class GitHubAuthenticationError(GitHubClientError):
    """Raised when authentication fails (401)."""
    pass


class GitHubClient:
    """
    Async client for interacting with the GitHub REST API.
    Handles authentication, error propagation, and specific repository interactions.
    """

    BASE_URL = "https://api.github.com"
    # Strict validation pattern for GitHub owner/repo names
    # Only allows alphanumerics, hyphens, underscores, and periods.
    VALID_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

    def __init__(self, token: str = settings.github_token, timeout_seconds: int = 30):
        """
        Initialize the GitHub client with a single reusable httpx.AsyncClient.
        """
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Repository-Intelligence-Agent/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        self.client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True
        )
        self.cache = _github_cache

    def _validate_owner_repo(self, owner: str, repo: str) -> None:
        """
        Strict validation to ensure owner and repo names are completely safe
        and do not contain path traversal vectors or invalid characters.
        """
        if not owner or not repo:
            raise ValueError("Owner and repo must not be empty.")
        if not owner.strip() or not repo.strip():
            raise ValueError("Owner and repo must not be whitespace-only.")
        if "/" in owner or "/" in repo:
            raise ValueError("Owner and repo must not contain slashes.")
        if ".." in owner or ".." in repo:
            raise ValueError("Path traversal sequences are not allowed.")
        if not self.VALID_NAME_PATTERN.match(owner) or not self.VALID_NAME_PATTERN.match(repo):
            raise ValueError("Owner and repo contain invalid characters.")

    async def _make_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """
        Centralized request method that handles error translation and safe network execution.
        """
        try:
            response = await self.client.request(method, path, **kwargs)
            
            # Handle specific GitHub API errors
            if response.status_code == 401:
                raise GitHubAuthenticationError("Unauthorized. Check your GitHub token.")
            elif response.status_code == 403:
                # Check rate limits explicitly from headers
                remaining = response.headers.get("x-ratelimit-remaining")
                if remaining is not None and int(remaining) <= 0:
                    raise GitHubRateLimitError("GitHub API rate limit exhausted based on headers.")
                # Fallback detection for rate limit block if headers didn't catch it
                if "rate limit" in response.text.lower():
                    raise GitHubRateLimitError("GitHub API rate limit exceeded.")
                raise GitHubClientError(f"Forbidden: {response.text}")
            elif response.status_code == 404:
                raise ResourceNotFoundError(f"Resource not found at {path}")
            
            response.raise_for_status()
            return response

        except httpx.TimeoutException as e:
            logger.error(f"GitHub API Timeout on {method} {path}")
            raise GitHubClientError("GitHub API request timed out.") from e
        except httpx.RequestError as e:
            logger.error(f"GitHub API Network Error: {e}")
            raise GitHubClientError(f"Network error communicating with GitHub: {e}") from e
        except httpx.HTTPStatusError as e:
            # Re-raise generic HTTP errors if not caught above
            raise GitHubClientError(f"GitHub API returned HTTP error: {e.response.status_code}") from e

    async def get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Fetch core repository metadata.
        """
        self._validate_owner_repo(owner, repo)
        path = f"/repos/{owner}/{repo}"
        
        try:
            response = await self._make_request("GET", path)
            data = response.json()
            
            return {
                "name": data.get("name", repo),
                "owner": owner,
                "repo": repo,
                "default_branch": data.get("default_branch", "main"),
                "description": data.get("description", ""),
                "html_url": data.get("html_url", ""),
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
            }
        except ResourceNotFoundError:
            # A 404 on the core repository path explicitly means the repo doesn't exist
            raise RepositoryNotFoundError(f"Repository {owner}/{repo} not found.")

    async def get_default_branch(self, owner: str, repo: str) -> str:
        """
        Utility to fetch strictly the default branch.
        """
        repo_data = await self.get_repository(owner, repo)
        return repo_data["default_branch"]

    async def get_readme(self, owner: str, repo: str) -> Optional[str]:
        """
        Fetch and decode the repository's root README.
        Returns the markdown text, or None if no README exists.
        """
        # First ensure the repo actually exists (raises RepositoryNotFoundError if not)
        # This properly distinguishes between a missing repo and a missing README.
        await self.get_repository(owner, repo)

        path = f"/repos/{owner}/{repo}/readme"
        
        try:
            response = await self._make_request("GET", path)
            data = response.json()
            
            content_base64 = data.get("content", "")
            if not content_base64:
                return None
                
            decoded_bytes = base64.b64decode(content_base64)
            return decoded_bytes.decode("utf-8", errors="replace")
            
        except ResourceNotFoundError:
            logger.info(f"No README found for {owner}/{repo}")
            return None

    async def get_code_and_docs(self, owner: str, repo: str) -> List[Dict[str, str]]:
        """
        Uses the Git Trees API to recursively discover markdown and source code documents.
        Filters for supported extensions.
        Excludes README.md.
        
        Returns a list of dictionaries containing path, html_url, and the fetched content.
        """
        self._validate_owner_repo(owner, repo)
        
        default_branch = await self.get_default_branch(owner, repo)
        
        # NOTE: Using the branch name directly in the Git Trees API is highly efficient 
        # for our MVP requirements. While branch->commit->tree SHA resolution prevents 
        # theoretical race conditions if the branch updates during fetching, branch-name 
        # lookup is fully acceptable and minimizes API calls for our one-shot ingestion MVP.
        tree_path = f"/repos/{owner}/{repo}/git/trees/{default_branch}"
        
        try:
            response = await self._make_request("GET", tree_path, params={"recursive": "1"})
        except ResourceNotFoundError:
            return []

        tree_data = response.json()
        
        if tree_data.get("truncated"):
            logger.warning(f"Git Trees API returned truncated results for {owner}/{repo}. Some documents may be missing.")
        
        documents = []
        base_html_url = f"https://github.com/{owner}/{repo}/blob/{default_branch}"
        base_raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}"
        
        for item in tree_data.get("tree", []):
            if item.get("type") != "blob":
                continue
                
            path: str = item.get("path", "")
            lower_path = path.lower()
            
            valid_extensions = (".md", ".py", ".ts", ".js", ".tsx", ".jsx", ".java", ".go", ".rs", ".cpp", ".c", ".h")
            if not lower_path.endswith(valid_extensions):
                continue
                
            if lower_path == "readme.md":
                continue
            
            # Fetch the raw content
            download_url = f"{base_raw_url}/{path}"
            try:
                # Use the client to fetch raw content (handling redirects/auth automatically if needed)
                content_resp = await self._make_request("GET", download_url)
                content = content_resp.text
            except Exception as e:
                logger.warning(f"Failed to download content for {path}: {e}")
                continue

            documents.append({
                "path": path,
                "html_url": f"{base_html_url}/{path}",
                "content": content
            })
                
        return documents

    async def get_contents(self, owner: str, repo: str, path: str) -> Dict[str, Any]:
        """
        Fetch file or directory contents from the GitHub Contents API.
        Results are cached to minimize API usage.
        """
        self._validate_owner_repo(owner, repo)
        cache_key = f"{owner}/{repo}:{path}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        api_path = f"/repos/{owner}/{repo}/contents/{path}"
        try:
            response = await self._make_request("GET", api_path)
            data = response.json()
            
            # If it's a file and base64 encoded, decode it for the LLM
            if isinstance(data, dict) and data.get("type") == "file" and data.get("encoding") == "base64":
                content_base64 = data.get("content", "")
                if content_base64:
                    try:
                        decoded_bytes = base64.b64decode(content_base64)
                        data["decoded_content"] = decoded_bytes.decode("utf-8", errors="replace")
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 content for {path}: {e}")
                        
            self.cache[cache_key] = data
            return data
            
        except ResourceNotFoundError:
            logger.info(f"Path not found: {path} in {owner}/{repo}")
            return {"error": f"Path not found: {path}"}

    async def get_repository_tree_cache(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Fetch the entire repository tree structure recursively.
        Cached heavily to serve as the primary tool for repository-wide questions.
        """
        self._validate_owner_repo(owner, repo)
        cache_key = f"{owner}/{repo}:tree"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        default_branch = await self.get_default_branch(owner, repo)
        tree_path = f"/repos/{owner}/{repo}/git/trees/{default_branch}"
        
        try:
            response = await self._make_request("GET", tree_path, params={"recursive": "1"})
            data = response.json()
            
            # Simplify the tree structure for the LLM context to save tokens
            simplified_tree = []
            for item in data.get("tree", []):
                simplified_tree.append({
                    "path": item.get("path"),
                    "type": item.get("type")
                })
                
            result = {"tree": simplified_tree, "truncated": data.get("truncated", False)}
            self.cache[cache_key] = result
            return result
        except ResourceNotFoundError:
            return {"error": f"Repository tree not found for {owner}/{repo}"}

    async def get_issues(self, owner: str, repo: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch open and closed issues, explicitly excluding pull requests.
        Paginates up to the specified hard limit.
        """
        self._validate_owner_repo(owner, repo)
        
        issues = []
        url = f"/repos/{owner}/{repo}/issues?state=all&per_page=100"
        
        while url and len(issues) < limit:
            try:
                response = await self._make_request("GET", url)
            except ResourceNotFoundError:
                break
                
            data = response.json()
            for item in data:
                # GitHub issues endpoint returns both issues and PRs.
                # PRs contain a "pull_request" key.
                if "pull_request" in item:
                    continue
                    
                issues.append({
                    "id": item.get("id"),
                    "number": item.get("number"),
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                    "html_url": item.get("html_url", ""),
                    "state": item.get("state", ""),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                })
                
                if len(issues) >= limit:
                    break
            
            # httpx automatically parses the Link header into response.links
            url = response.links.get("next", {}).get("url")
            
        return issues

    async def get_pull_requests(self, owner: str, repo: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch open and closed pull requests.
        Paginates up to the specified hard limit.
        """
        self._validate_owner_repo(owner, repo)
        
        prs = []
        url = f"/repos/{owner}/{repo}/pulls?state=all&per_page=100"
        
        while url and len(prs) < limit:
            try:
                response = await self._make_request("GET", url)
            except ResourceNotFoundError:
                break
                
            data = response.json()
            for item in data:
                prs.append({
                    "id": item.get("id"),
                    "number": item.get("number"),
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                    "html_url": item.get("html_url", ""),
                    "state": item.get("state", ""),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                })
                
                if len(prs) >= limit:
                    break
            
            url = response.links.get("next", {}).get("url")
            
        return prs

    async def close(self) -> None:
        """
        Properly close the underlying HTTP client.
        """
        await self.client.aclose()

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Any) -> None:
        await self.close()
