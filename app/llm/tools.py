from typing import List
from langchain_core.tools import tool
from app.github.github_client import GitHubClient
import logging

logger = logging.getLogger(__name__)

def get_github_tools(repository_id: str):
    """
    Factory function to create tools bound to a specific repository.
    This prevents the LLM from hallucinating or browsing other repositories.
    """
    try:
        owner, repo = repository_id.split("/")
    except ValueError:
        logger.error(f"Invalid repository_id format: {repository_id}")
        owner, repo = "", ""
    
    @tool
    async def list_github_directory(path: str) -> str:
        """Returns a list of files and folders in a specific directory path of the repository."""
        if not owner or not repo:
            return "Error: Invalid repository context."
            
        async with GitHubClient() as client:
            data = await client.get_contents(owner, repo, path)
            if isinstance(data, dict) and "error" in data:
                return data["error"]
            if isinstance(data, list):
                result = []
                for item in data:
                    result.append(f"{item.get('type')}: {item.get('path')}")
                return "\n".join(result)
            return "Path is a file, not a directory."

    @tool
    async def read_github_file(path: str) -> str:
        """Returns the content of a specific file in the repository."""
        if not owner or not repo:
            return "Error: Invalid repository context."
            
        async with GitHubClient() as client:
            data = await client.get_contents(owner, repo, path)
            if isinstance(data, dict):
                if "error" in data:
                    return data["error"]
                if "decoded_content" in data:
                    return data["decoded_content"]
                if data.get("type") == "dir":
                    return "Path is a directory, not a file."
            return "Failed to read file."

    @tool
    async def get_repository_tree() -> str:
        """Returns the full repository structure (all files and directories). Use this to understand project architecture and list all files."""
        if not owner or not repo:
            return "Error: Invalid repository context."
            
        async with GitHubClient() as client:
            data = await client.get_repository_tree_cache(owner, repo)
            if "error" in data:
                return data["error"]
            tree = data.get("tree", [])
            lines = [f"{item.get('type')}: {item.get('path')}" for item in tree]
            if data.get("truncated"):
                lines.append("... (truncated)")
            return "\n".join(lines)

    return [list_github_directory, read_github_file, get_repository_tree]
