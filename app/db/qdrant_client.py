import logging
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from app.schemas.config import settings
from app.schemas.embeddings import EmbeddingOutput

logger = logging.getLogger(__name__)

COLLECTION_NAME = "repository_chunks"
VECTOR_SIZE = 384
DISTANCE_METRIC = models.Distance.COSINE

class QdrantService:
    def __init__(self):
        try:
            self.client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=30.0
            )
        except Exception as e:
            logger.error(f"Failed to initialize QdrantClient: {e}", exc_info=True)
            raise

    async def initialize_collection(self) -> None:
        """
        Idempotent startup hook to ensure the `repository_chunks` collection exists 
        with the correct vector size and distance metric.
        """
        try:
            collection_exists = await self.client.collection_exists(COLLECTION_NAME)
            if collection_exists:
                logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists. Skipping creation.")
                return

            logger.info(f"Creating Qdrant collection '{COLLECTION_NAME}' (Size: {VECTOR_SIZE}, Metric: {DISTANCE_METRIC})...")
            await self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=DISTANCE_METRIC
                )
            )
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' created successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection '{COLLECTION_NAME}': {e}", exc_info=True)
            # Raise exception so the startup hook fails cleanly, stopping FastAPI if Qdrant is completely broken
            raise

    async def close(self) -> None:
        """Close the underlying client connection."""
        await self.client.close()

    async def get_repository_summary(self, repository_id: str) -> dict[str, int]:
        """
        Executes parallel count queries against Qdrant to compute chunk statistics
        grouped by source_type for a specific repository.
        """
        import asyncio
        
        async def count_type(source_type: str) -> int:
            try:
                response = await self.client.count(
                    collection_name=COLLECTION_NAME,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="repository_id",
                                match=models.MatchValue(value=repository_id)
                            ),
                            models.FieldCondition(
                                key="source_type",
                                match=models.MatchValue(value=source_type)
                            )
                        ]
                    )
                )
                return response.count
            except Exception as e:
                logger.error(f"Count query failed for {source_type}: {e}")
                return 0

        # Execute all 4 count queries concurrently for maximum efficiency
        readme_count, docs_count, issue_count, pr_count = await asyncio.gather(
            count_type("readme"),
            count_type("docs"),
            count_type("issue"),
            count_type("pull_request")
        )
        
        return {
            "readme_chunks": readme_count,
            "documentation_chunks": docs_count,
            "issue_chunks": issue_count,
            "pull_request_chunks": pr_count,
            "total_chunks": readme_count + docs_count + issue_count + pr_count
        }

    async def store_embeddings(self, embedding_output: 'EmbeddingOutput', batch_size: int = 128) -> dict:
        """
        Takes embedded chunks and upserts them into Qdrant efficiently using batched execution.
        """
        import uuid
        
        points = []
        repository_id = None
        
        for embedded_chunk in embedding_output.embeddings:
            if not repository_id:
                repository_id = embedded_chunk.metadata.repository_id
                
            # Qdrant strictly requires UUID string or unsigned int for IDs.
            # We hash the deterministic chunk_id into a valid UUID string format.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, embedded_chunk.chunk_id))
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedded_chunk.vector,
                    payload={
                        "text": embedded_chunk.text,
                        "source_type": embedded_chunk.source_type,
                        "title": embedded_chunk.title,
                        "url": embedded_chunk.metadata.url,
                        "repository_id": embedded_chunk.metadata.repository_id
                    }
                )
            )
            
        if not points:
            return {"status": "success", "inserted_count": 0, "repository_id": None}

        try:
            # We use chunks of points if the array exceeds batch_size
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )
                
            logger.info(f"Successfully upserted {len(points)} vectors into {COLLECTION_NAME} for {repository_id}")
            return {
                "status": "success",
                "inserted_count": len(points),
                "repository_id": repository_id
            }
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors into Qdrant for {repository_id}: {e}", exc_info=True)
            raise

# Provide a singleton instance
qdrant_service = QdrantService()
