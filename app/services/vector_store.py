"""
Vector store service for managing vector operations
"""

import os
from typing import List, Dict, Any, Optional
from app.models.vector_store import VectorStoreConfig, EmbeddingConfig
from app.vector.pinecone_store import PineconeVectorStore
from app.vector.pinecone_inference_store import PineconeInferenceVectorStore
from app.core.logging import get_logger

logger = get_logger(__name__)

class VectorStoreService:
    """Service for managing vector store operations"""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize the vector store service"""
        self.config = config
        
        # Choose the appropriate vector store implementation
        if self.config.use_inference:
            self.vector_store = PineconeInferenceVectorStore(config)
            logger.info(f"Using PineconeInferenceVectorStore with model: {config.inference_model}")
        else:
            self.vector_store = PineconeVectorStore(config)
            logger.info("Using PineconeVectorStore")
            
    async def initialize(self):
        """Initialize the vector store"""
        await self.vector_store.initialize()
        
    async def cleanup(self):
        """Clean up resources"""
        await self.vector_store.cleanup()
        
    async def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """Add vectors to the store"""
        await self.vector_store.add_vectors(vectors, metadata, ids)
        
    async def search(self, query_vector: List[float], top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None):
        """Search for similar vectors"""
        return await self.vector_store.search(query_vector, top_k, filter_metadata)
        
    async def delete_vectors(self, ids: List[str]):
        """Delete vectors from the store"""
        await self.vector_store.delete_vectors(ids)
        
    async def get_stats(self):
        """Get statistics about the vector store"""
        return await self.vector_store.get_stats() 