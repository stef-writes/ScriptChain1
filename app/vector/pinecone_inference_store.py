"""
Pinecone vector store implementation with native inference
"""

from typing import List, Dict, Any, Optional, Tuple
import pinecone
import numpy as np
import logging
from datetime import datetime
from app.models.vector_store import (
    VectorStoreConfig,
    VectorSearchResult,
    BatchOperation
)
from app.vector.base import (
    VectorStoreInterface,
    VectorStoreError,
    with_retry,
    BatchProcessor
)

logger = logging.getLogger(__name__)

class PineconeInferenceVectorStore(VectorStoreInterface):
    """Vector store implementation using Pinecone with native inference"""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize Pinecone vector store with inference capabilities.
        
        Args:
            config: Vector store configuration
        """
        super().__init__(config)
        self.index = None
        self.pinecone = None
        self.batch_processor = BatchProcessor(config.batch_size)
        
    async def initialize(self) -> None:
        """Initialize Pinecone resources"""
        try:
            # Initialize Pinecone
            init_params = {
                "api_key": self.config.api_key,
                "environment": self.config.environment
            }
            
            # Add host if provided
            if self.config.host:
                init_params["host"] = self.config.host
                logger.info(f"Using custom Pinecone host: {self.config.host}")
            
            self.pinecone = pinecone.Pinecone(**init_params)
            
            # Create index if it doesn't exist
            if self.config.index_name not in [index.name for index in self.pinecone.list_indexes()]:
                self.pinecone.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric.value,
                    spec=pinecone.IndexSpec(
                        pod_type=self.config.pod_type,
                        pods=self.config.replicas,
                        metadata_config=self.config.metadata_config
                    )
                )
                logger.info(f"Created new Pinecone index: {self.config.index_name}")
            
            # Connect to index
            self.index = self.pinecone.Index(self.config.index_name)
            logger.info(f"Connected to Pinecone index: {self.config.index_name}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pinecone: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup Pinecone resources"""
        if self.index:
            try:
                # No explicit cleanup needed for Pinecone
                self.index = None
                logger.info("Cleaned up Pinecone resources")
            except Exception as e:
                logger.error(f"Error during Pinecone cleanup: {str(e)}")
    
    def _validate_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Validate vectors before adding to store.
        
        Args:
            vectors: List of vectors
            metadata: List of metadata dictionaries
            ids: Optional list of vector IDs
            
        Raises:
            VectorStoreError: If validation fails
        """
        if not vectors:
            raise VectorStoreError("No vectors provided")
            
        if len(vectors) != len(metadata):
            raise VectorStoreError(
                f"Number of vectors ({len(vectors)}) does not match "
                f"number of metadata items ({len(metadata)})"
            )
            
        if ids and len(ids) != len(vectors):
            raise VectorStoreError(
                f"Number of IDs ({len(ids)}) does not match "
                f"number of vectors ({len(vectors)})"
            )
            
        for i, vector in enumerate(vectors):
            if len(vector) != self.config.dimension:
                raise VectorStoreError(
                    f"Vector {i} has incorrect dimension "
                    f"(expected {self.config.dimension}, got {len(vector)})"
                )
    
    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Pinecone.
        
        Args:
            vectors: List of vectors to add
            metadata: List of metadata dictionaries
            ids: Optional list of vector IDs
            
        Raises:
            VectorStoreError: If operation fails
        """
        if not self.index:
            raise VectorStoreError("Vector store not initialized")
            
        self._validate_vectors(vectors, metadata, ids)
        
        # Generate IDs if not provided
        if not ids:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            ids = [f"vec_{timestamp}_{i}" for i in range(len(vectors))]
        
        # Prepare vectors for upsert
        vector_data = [
            (id, vector, meta)
            for id, vector, meta in zip(ids, vectors, metadata)
        ]
        
        # Define batch operation
        async def upsert_batch(batch: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
            batch_ids = [b[0] for b in batch]
            batch_vectors = [b[1] for b in batch]
            batch_metadata = [b[2] for b in batch]
            
            await with_retry(
                lambda: self.index.upsert(
                    vectors=[{
                        'id': id,
                        'values': vector,
                        'metadata': meta
                    } for id, vector, meta in zip(batch_ids, batch_vectors, batch_metadata)]
                )
            )
        
        try:
            await self.batch_processor.process(vector_data, upsert_batch)
            logger.info(f"Added {len(vectors)} vectors to Pinecone")
        except Exception as e:
            raise VectorStoreError(f"Failed to add vectors: {str(e)}")
    
    async def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "default"
    ) -> None:
        """Add texts to Pinecone using native inference.
        
        Args:
            texts: List of texts to embed and add
            ids: Optional list of IDs
            metadata: Optional list of metadata dictionaries
            namespace: Namespace to add vectors to
            
        Raises:
            VectorStoreError: If operation fails
        """
        if not self.pinecone or not self.index:
            raise VectorStoreError("Vector store not initialized")
            
        if not texts:
            raise VectorStoreError("No texts provided")
            
        # Generate IDs if not provided
        if not ids:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            ids = [f"text_{timestamp}_{i}" for i in range(len(texts))]
            
        # Generate metadata if not provided
        if not metadata:
            metadata = [{"text": text} for text in texts]
            
        # Get embeddings using Pinecone inference
        embeddings = self.pinecone.inference.embed(
            model=self.config.inference_model,
            inputs=texts,
            parameters={
                "input_type": "passage"
            }
        )
        
        # Prepare vectors for upsert
        vectors = []
        for i, (id, embedding, meta) in enumerate(zip(ids, embeddings, metadata)):
            vectors.append({
                "id": id,
                "values": embedding['values'],
                "metadata": meta
            })
            
        # Define batch operation
        async def upsert_batch(batch: List[Dict[str, Any]]) -> None:
            await with_retry(
                lambda: self.index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
            )
        
        try:
            await self.batch_processor.process(vectors, upsert_batch)
            logger.info(f"Added {len(texts)} texts to Pinecone")
        except Exception as e:
            raise VectorStoreError(f"Failed to add texts: {str(e)}")
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in Pinecone.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results
            
        Raises:
            VectorStoreError: If operation fails
        """
        if not self.index:
            raise VectorStoreError("Vector store not initialized")
            
        if len(query_vector) != self.config.dimension:
            raise VectorStoreError(
                f"Query vector has incorrect dimension "
                f"(expected {self.config.dimension}, got {len(query_vector)})"
            )
        
        try:
            # Perform search
            results = await with_retry(
                lambda: self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_values=True,
                    include_metadata=True,
                    filter=filter_metadata
                )
            )
            
            # Convert results to VectorSearchResult objects
            search_results = []
            for match in results.matches:
                search_results.append(
                    VectorSearchResult(
                        id=match.id,
                        score=float(match.score),
                        vector=match.values,
                        metadata=match.metadata
                    )
                )
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search vectors: {str(e)}")
    
    async def search_text(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "default"
    ) -> List[VectorSearchResult]:
        """Search for similar texts in Pinecone.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            namespace: Namespace to search in
            
        Returns:
            List of search results
            
        Raises:
            VectorStoreError: If operation fails
        """
        if not self.pinecone or not self.index:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            # Get embedding using Pinecone inference
            embedding = self.pinecone.inference.embed(
                model=self.config.inference_model,
                inputs=[query],
                parameters={
                    "input_type": "query"
                }
            )
            
            # Perform search
            results = await with_retry(
                lambda: self.index.query(
                    vector=embedding[0]['values'],
                    top_k=top_k,
                    include_values=False,
                    include_metadata=True,
                    filter=filter_metadata,
                    namespace=namespace
                )
            )
            
            # Convert results to VectorSearchResult objects
            search_results = []
            for match in results.matches:
                search_results.append(
                    VectorSearchResult(
                        id=match.id,
                        score=float(match.score),
                        metadata=match.metadata
                    )
                )
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search texts: {str(e)}")
    
    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Pinecone.
        
        Args:
            ids: List of vector IDs to delete
            
        Raises:
            VectorStoreError: If operation fails
        """
        if not self.index:
            raise VectorStoreError("Vector store not initialized")
            
        if not ids:
            raise VectorStoreError("No vector IDs provided")
        
        # Define batch operation
        async def delete_batch(batch_ids: List[str]) -> None:
            await with_retry(
                lambda: self.index.delete(ids=batch_ids)
            )
        
        try:
            await self.batch_processor.process(ids, delete_batch)
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
        except Exception as e:
            raise VectorStoreError(f"Failed to delete vectors: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary containing statistics
            
        Raises:
            VectorStoreError: If operation fails
        """
        if not self.index:
            raise VectorStoreError("Vector store not initialized")
            
        try:
            stats = await with_retry(
                lambda: self.index.describe_index_stats()
            )
            return {
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "total_vector_count": stats.total_vector_count,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            raise VectorStoreError(f"Failed to get stats: {str(e)}") 