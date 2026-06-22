import logging
from app.github.github_client import GitHubClient
from app.schemas.chunking import DocumentInput, DocumentMetadata
from app.schemas.embeddings import EmbeddingInput
from app.utils.chunking import document_chunker
from app.embeddings.dense import dense_embedding_model
from app.db.qdrant_client import qdrant_service

logger = logging.getLogger(__name__)

class IngestionService:
    @staticmethod
    async def process_repository(owner: str, repo: str) -> None:
        """
        Background task to orchestrate full repository ingestion:
        Fetch -> Chunk -> Embed -> Qdrant Upsert
        """
        repository_id = f"{owner}/{repo}"
        try:
            logger.info(f"Starting ingestion for {repository_id}")
            
            all_chunks = []
            
            async with GitHubClient() as client:
                metadata = await client.get_repository(owner, repo)
                repo_url = metadata.get('html_url', f"https://github.com/{repository_id}")
                
                # Helper to process and chunk documents
                def process_content(source_type: str, title: str, content: str, url: str):
                    if not content: return
                    doc_input = DocumentInput(
                        source_type=source_type,
                        title=title,
                        content=content,
                        metadata=DocumentMetadata(url=url, repository_id=repository_id)
                    )
                    chunks = document_chunker.chunk_document(doc_input)
                    all_chunks.extend(chunks)

                # Process README
                readme = await client.get_readme(owner, repo)
                if readme:
                    process_content("readme", "README", readme, f"{repo_url}#readme")
                
                # Process Code and Docs
                code_and_docs = await client.get_code_and_docs(owner, repo)
                for doc in code_and_docs:
                    process_content("docs", doc["path"], doc.get("content", ""), doc["html_url"])
                
                # Process Issues
                issues = await client.get_issues(owner, repo)
                for issue in issues:
                    title = f"Issue #{issue['number']}: {issue['title']}"
                    process_content("issue", title, issue["body"] or issue["title"], issue["html_url"])
                
                # Process PRs
                prs = await client.get_pull_requests(owner, repo)
                for pr in prs:
                    title = f"PR #{pr['number']}: {pr['title']}"
                    process_content("pull_request", title, pr["body"] or pr["title"], pr["html_url"])
                
            if not all_chunks:
                logger.warning(f"No chunks generated for {repository_id}. Skipping embedding.")
                return
                
            logger.info(f"Generated {len(all_chunks)} total chunks for {repository_id}. Starting embedding...")
            
            # Embed all chunks
            embedding_input = EmbeddingInput(chunks=all_chunks)
            embedded_output = dense_embedding_model.encode_batch(embedding_input)
            
            logger.info(f"Generated {len(embedded_output.embeddings)} embeddings. Upserting to Qdrant...")
            
            # Store in Qdrant
            result = await qdrant_service.store_embeddings(embedded_output)
            
            logger.info(f"Successfully completed ingestion for {repository_id}: inserted {result['inserted_count']} vectors.")
            
        except Exception as e:
            logger.error(f"Ingestion failed for {repository_id}: {str(e)}", exc_info=True)
