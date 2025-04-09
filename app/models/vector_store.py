"""
Configuration models for vector store operations
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"

class VectorStoreConfig(BaseModel):
    """Configuration for vector store"""
    index_name: str = Field(..., description="Name of the vector store index")
    environment: str = Field(..., description="Environment name")
    dimension: int = Field(..., description="Dimension of vectors")
    metric: SimilarityMetric = Field(..., description="Distance metric for similarity search")
    pod_type: str = Field(..., description="Type of pod")
    replicas: int = Field(..., description="Number of replicas")
    use_inference: bool = Field(..., description="Whether to use inference")
    inference_model: str = Field(..., description="Model to use for inference")
    api_key: str = Field(..., description="API key")
    host: Optional[str] = Field(None, description="Host URL for the vector store")
    metadata_config: Dict[str, str] = Field(
        default_factory=dict,
        description="Metadata field configuration"
    )
    batch_size: int = Field(100, description="Batch size for operations")

class EmbeddingConfig(BaseModel):
    """Configuration for embedding operations"""
    model_name: str = Field(
        "all-MiniLM-L6-v2",
        description="Name of the sentence transformer model"
    )
    batch_size: int = Field(32, description="Batch size for embedding operations")
    cache_size: int = Field(1000, description="Size of embedding cache")
    max_length: Optional[int] = Field(512, description="Maximum sequence length")
    normalize_embeddings: bool = Field(True, description="Whether to normalize embeddings")

class VectorSearchResult(BaseModel):
    """Result from vector search operation"""
    id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None

class BatchOperation(BaseModel):
    """Batch operation for vector store"""
    vectors: List[List[float]]
    ids: List[str]
    metadata: List[Dict[str, Any]] 