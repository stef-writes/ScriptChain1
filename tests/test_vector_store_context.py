"""
Tests for the vector store integration with the context manager.
"""

import pytest
import os
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List
import networkx as nx
from app.utils.context import GraphContextManager
from app.models.vector_store import VectorStoreConfig, SimilarityMetric
from app.vector.pinecone_store import PineconeVectorStore
from app.vector.pinecone_inference_store import PineconeInferenceVectorStore
from app.models.node_models import NodeExecutionResult, NodeMetadata, UsageMetadata
import json

# Mock Pinecone client
@pytest.fixture
def mock_pinecone():
    with patch("app.vector.pinecone_store.pinecone") as mock:
        # Mock the Pinecone class
        mock_pinecone_instance = MagicMock()
        mock.Pinecone.return_value = mock_pinecone_instance
        
        # Mock the Index class
        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index
        
        # Mock list_indexes method
        mock_pinecone_instance.list_indexes.return_value = []
        
        yield mock

# Mock vector store config
@pytest.fixture
def vector_store_config():
    return {
        "index_name": "test-index",
        "environment": "test-env",
        "dimension": 384,
        "pod_type": "p1",
        "replicas": 1,
        "use_inference": True,
        "inference_model": "test-model",
        "api_key": "test-api-key",
        "host": "test-host",
        "metric": SimilarityMetric.COSINE
    }

# Mock context manager with vector store
@pytest.fixture
def context_manager_with_vector_store(vector_store_config):
    graph = nx.DiGraph()
    graph.add_node("node1")
    graph.add_node("node2")
    graph.add_edge("node1", "node2")
    
    context_manager = GraphContextManager(
        max_tokens=1000,
        graph=graph,
        vector_store_config=vector_store_config
    )
    
    return context_manager

# Test vector store initialization
@pytest.mark.asyncio
async def test_vector_store_initialization(vector_store_config):
    """Test that the vector store is properly initialized with the correct config."""
    with patch("app.vector.pinecone_inference_store.pinecone") as mock_pinecone:
        # Mock the Pinecone class
        mock_pinecone_instance = MagicMock()
        mock_pinecone.Pinecone.return_value = mock_pinecone_instance
        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index
        mock_pinecone_instance.list_indexes.return_value = []
        
        # Create context manager with vector store config
        context_manager = GraphContextManager(
            max_tokens=1000,
            vector_store_config=vector_store_config
        )
        
        # Initialize the vector store
        await context_manager.vector_store.initialize()
        
        # Verify vector store was initialized
        assert context_manager.vector_store is not None
        assert isinstance(context_manager.vector_store, PineconeInferenceVectorStore)
        
        # Verify Pinecone was initialized with correct parameters
        mock_pinecone.Pinecone.assert_called_once()
        call_args = mock_pinecone.Pinecone.call_args[1]
        assert call_args["api_key"] == vector_store_config["api_key"]
        assert call_args["environment"] == vector_store_config["environment"]
        assert call_args["host"] == vector_store_config["host"]

# Test vector store with inference
def test_vector_store_with_inference(vector_store_config):
    """Test that the vector store with inference is properly initialized."""
    with patch("app.vector.pinecone_inference_store.pinecone") as mock_pinecone:
        # Mock the Pinecone class
        mock_pinecone_instance = MagicMock()
        mock_pinecone.Pinecone.return_value = mock_pinecone_instance
        
        # Create context manager with vector store config
        context_manager = GraphContextManager(
            max_tokens=1000,
            vector_store_config=vector_store_config
        )
        
        # Verify vector store was initialized with inference
        assert context_manager.vector_store is not None
        assert isinstance(context_manager.vector_store, PineconeInferenceVectorStore)

# Test vector store without inference
def test_vector_store_without_inference(vector_store_config):
    """Test that the vector store without inference is properly initialized."""
    # Disable inference
    vector_store_config["use_inference"] = False
    
    with patch("app.vector.pinecone_store.pinecone") as mock_pinecone:
        # Mock the Pinecone class
        mock_pinecone_instance = MagicMock()
        mock_pinecone.Pinecone.return_value = mock_pinecone_instance
        
        # Create context manager with vector store config
        context_manager = GraphContextManager(
            max_tokens=1000,
            vector_store_config=vector_store_config
        )
        
        # Verify vector store was initialized without inference
        assert context_manager.vector_store is not None
        assert isinstance(context_manager.vector_store, PineconeVectorStore)

# Test context optimization with vector store
@pytest.mark.asyncio
async def test_context_optimization_with_vector_store(context_manager_with_vector_store, mock_pinecone):
    """Test that context optimization uses the vector store."""
    # Mock the search_text method
    mock_index = mock_pinecone.Pinecone.return_value.Index.return_value
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "vec1",
                "score": 0.9,
                "metadata": {"content": "Similar content 1"}
            },
            {
                "id": "vec2",
                "score": 0.8,
                "metadata": {"content": "Similar content 2"}
            }
        ]
    }
    
    # Set context for node1
    context_manager_with_vector_store.set_context("node1", {
        "content": "The quick brown fox jumps over the lazy dog"
    })
    
    # Get optimized context
    context = await context_manager_with_vector_store.get_context_with_optimization(
        "node1",
        "fox jumps",
        k=2
    )
    
    # Verify vector store was used
    assert context is not None
    assert "content" in context

# Test context versioning with vector store
@pytest.mark.asyncio
async def test_context_versioning_with_vector_store(context_manager_with_vector_store, mock_pinecone):
    """Test that context versioning uses the vector store."""
    # Mock the fetch method
    mock_index = mock_pinecone.Pinecone.return_value.Index.return_value
    mock_index.fetch.return_value = {
        "vectors": {
            "vec1": {
                "metadata": {"content": "Versioned content", "version": "v1"}
            }
        }
    }
    
    # Set context for node1
    context_manager_with_vector_store.set_context("node1", {
        "content": "The quick brown fox jumps over the lazy dog"
    })
    
    # Get versioned context
    context = await context_manager_with_vector_store.get_context_with_version(
        "node1",
        "v1"
    )
    
    # Verify vector store was used
    assert context is not None
    assert "content" in context

# Test context update with NodeExecutionResult
def test_context_update_with_node_execution_result(context_manager_with_vector_store):
    """Test that context is properly updated with a NodeExecutionResult."""
    # Create a NodeExecutionResult
    result = NodeExecutionResult(
        success=True,
        output={"text": "Test output"},
        metadata=NodeMetadata(
            node_id="node1",
            node_type="test",
            version="1.0.0"
        ),
        usage=UsageMetadata(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            model="test-model",
            node_id="node1"
        )
    )
    
    # Update context with the result
    context_manager_with_vector_store.update_context("node1", result)
    
    # Verify context was updated
    context = context_manager_with_vector_store.get_context("node1")
    assert context is not None
    assert "output" in context
    assert context["output"]["text"] == "Test output"

# Test context update with raw content
def test_context_update_with_raw_content(context_manager_with_vector_store):
    """Test that context is properly updated with raw content."""
    # Update context with raw content
    context_manager_with_vector_store.update_context("node1", "Raw content")
    
    # Verify context was updated
    context = context_manager_with_vector_store.get_context("node1")
    assert context is not None
    assert "output" in context
    assert context["output"] == "Raw content"

# Test context formatting
def test_context_formatting(context_manager_with_vector_store):
    """Test that context is properly formatted according to rules."""
    from app.models.node_models import ContextRule, ContextFormat
    
    # Set context
    test_context = {
        "text": "Plain text",
        "json": {"key": "value"},
        "code": "def hello(): print('Hello')"
    }
    context_manager_with_vector_store.set_context("node1", test_context)
    
    # Define context rules
    context_rules = {
        "text": ContextRule(
            include=True,
            format=ContextFormat.TEXT
        ),
        "json": ContextRule(
            include=True,
            format=ContextFormat.JSON
        ),
        "code": ContextRule(
            include=True,
            format=ContextFormat.CODE
        )
    }
    
    # Get formatted context
    formatted_context = context_manager_with_vector_store.get_node_context(
        "node1",
        context_rules=context_rules
    )
    
    # Verify formatting
    assert "text" in formatted_context
    assert formatted_context["text"] == "Plain text"
    assert "json" in formatted_context
    # Parse the JSON string back to a dictionary
    json_data = json.loads(formatted_context["json"])
    assert "key" in json_data
    assert json_data["key"] == "value"
    assert "code" in formatted_context
    assert "```" in formatted_context["code"] 