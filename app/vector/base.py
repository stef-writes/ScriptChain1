"""
Base classes and utilities for vector store operations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, TypeVar, Optional
import asyncio
import logging
from datetime import datetime
from app.models.vector_store import VectorStoreConfig, VectorSearchResult, BatchOperation

logger = logging.getLogger(__name__)

T = TypeVar('T')

class VectorStoreError(Exception):
    """Base exception for vector store operations"""
    pass

class VectorStoreBatchError(VectorStoreError):
    """Exception for batch operation failures"""
    def __init__(self, message: str, failed_items: List[int]):
        super().__init__(message)
        self.failed_items = failed_items

class VectorStoreTimeoutError(VectorStoreError):
    """Exception for timeout errors"""
    pass

class VectorStoreValidationError(VectorStoreError):
    """Exception for validation errors"""
    pass

async def with_retry(
    operation: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    timeout: float = 30.0
) -> T:
    """Execute an operation with exponential backoff retry.
    
    Args:
        operation: Async operation to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        timeout: Operation timeout in seconds
        
    Returns:
        Operation result
        
    Raises:
        VectorStoreError: If operation fails after all retries
        VectorStoreTimeoutError: If operation times out
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(operation(), timeout=timeout)
        except asyncio.TimeoutError:
            last_error = VectorStoreTimeoutError(
                f"Operation timed out after {timeout} seconds"
            )
            logger.warning(f"Attempt {attempt + 1}/{max_retries} timed out")
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            
        if attempt < max_retries - 1:
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)
    
    raise VectorStoreError(
        f"Operation failed after {max_retries} attempts"
    ) from last_error

class VectorStoreInterface(ABC):
    """Abstract base class for vector store implementations"""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize vector store.
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate vector store configuration"""
        if self.config.dimension <= 0:
            raise VectorStoreValidationError("Dimension must be positive")
        if self.config.batch_size <= 0:
            raise VectorStoreValidationError("Batch size must be positive")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize vector store resources"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup vector store resources"""
        pass
    
    @abstractmethod
    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to store.
        
        Args:
            vectors: List of vectors to add
            metadata: List of metadata dictionaries
            ids: Optional list of vector IDs
            
        Raises:
            VectorStoreError: If operation fails
            VectorStoreBatchError: If batch operation partially fails
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results
            
        Raises:
            VectorStoreError: If operation fails
        """
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from store.
        
        Args:
            ids: List of vector IDs to delete
            
        Raises:
            VectorStoreError: If operation fails
        """
        pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

class BatchProcessor:
    """Helper class for processing operations in batches"""
    
    def __init__(self, batch_size: int):
        """Initialize batch processor.
        
        Args:
            batch_size: Size of batches
        """
        self.batch_size = batch_size
        
    async def process(
        self,
        items: List[T],
        operation: Callable[[List[T]], Any]
    ) -> List[Any]:
        """Process items in batches.
        
        Args:
            items: Items to process
            operation: Batch operation to apply
            
        Returns:
            List of operation results
            
        Raises:
            VectorStoreBatchError: If any batch fails
        """
        results = []
        failed_items = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            try:
                batch_result = await operation(batch)
                results.extend(batch_result)
            except Exception as e:
                failed_items.extend(range(i, i + len(batch)))
                logger.error(f"Batch operation failed: {str(e)}")
        
        if failed_items:
            raise VectorStoreBatchError(
                f"Failed to process {len(failed_items)} items",
                failed_items
            )
            
        return results

class VectorStore:
    """Base class for vector store implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize vector store with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.index_name = config.get('index_name', 'default-index')
        self.environment = config.get('environment', 'us-west1')
        self.dimension = config.get('dimension', 1024)
        self.metric = config.get('metric', 'cosine')
        self.host = config.get('host')
        self.api_key = config.get('api_key')
        
        # Initialize store
        self._initialize()
        
    def _initialize(self):
        """Initialize the vector store"""
        logger.info(f"Initializing vector store: {self.index_name}")
        # This is a placeholder - actual implementations will override this
        
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add texts to the vector store
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of IDs for the added texts
        """
        logger.debug(f"Adding {len(texts)} texts to vector store")
        # This is a placeholder - actual implementations will override this
        return [f"id_{i}" for i in range(len(texts))]
        
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar texts
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter criteria
            
        Returns:
            List of similar texts with metadata
        """
        logger.debug(f"Searching for similar texts to: {query}")
        # This is a placeholder - actual implementations will override this
        return [{"text": f"result_{i}", "metadata": {}} for i in range(k)]
        
    async def delete(self, ids: List[str]) -> None:
        """Delete texts from the vector store
        
        Args:
            ids: List of IDs to delete
        """
        logger.debug(f"Deleting {len(ids)} texts from vector store")
        # This is a placeholder - actual implementations will override this 