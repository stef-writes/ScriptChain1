"""
Tests for the GraphContextManager implementation and its enhanced features.
"""

import pytest
import networkx as nx
from typing import Dict, Any, List, Optional
from app.utils.context import GraphContextManager
from app.vector.base import VectorStore, VectorStoreInterface
from app.models.node_models import (
    NodeExecutionResult, 
    NodeConfig, 
    NodeMetadata, 
    NodeIO,
    ContextRule,
    ContextFormat,
    UsageMetadata
)
from datetime import datetime
from app.models.vector_store import VectorStoreConfig, VectorSearchResult, SimilarityMetric
from unittest.mock import AsyncMock, MagicMock

class MockNodeExecutionResult(NodeExecutionResult):
    """Mock class for testing NodeExecutionResult functionality"""
    def __init__(self, output: str, success: bool = True, error: Optional[str] = None, node_id: str = "test_node", node_type: str = "test_type"):
        metadata = NodeMetadata(
            node_id=node_id,
            node_type=node_type,
            version="1.0.0",
            description="Test node"
        )
        usage = UsageMetadata(
            model="test_model",
            node_id=node_id,
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost=0.0
        )
        super().__init__(
            success=success,
            error=error,
            output={"text": output},
            metadata=metadata,
            usage=usage,
            execution_time=0.1,
            context_used={"test": "context"}
        )

class MockVectorStore(VectorStoreInterface):
    """Mock vector store for testing"""
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        
    async def initialize(self) -> None:
        pass
        
    async def cleanup(self) -> None:
        pass
        
    async def add_vectors(self, vectors, metadata, ids=None) -> None:
        pass
        
    async def search(self, query_vector, top_k, filter_metadata=None) -> List[VectorSearchResult]:
        return [
            VectorSearchResult(id="test1", score=0.9, metadata={"text": "test content"}),
            VectorSearchResult(id="test2", score=0.8, metadata={"text": "more content"})
        ]
        
    async def delete_vectors(self, ids) -> None:
        pass

    async def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        return [f"id_{i}" for i in range(len(texts))]

    async def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return [{"text": "test content", "metadata": {"node_id": "test_node"}} for _ in range(k)]

@pytest.fixture
def vector_store():
    config = VectorStoreConfig(
        index_name="test-index",
        environment="test",
        dimension=1536,
        metric=SimilarityMetric.COSINE,
        pod_type="p1",
        replicas=1,
        use_inference=True,
        inference_model="text-embedding-ada-002",
        api_key="test-key",
        batch_size=100
    )
    store = MockVectorStore(config)
    return store

@pytest.fixture
def test_graph():
    G = nx.DiGraph()
    
    node1_config = NodeConfig(
        id="node1",
        type="text_generation",
        model="gpt-4",
        prompt="test prompt",
        metadata=NodeMetadata(
            node_id="node1",
            node_type="text_generation",
            description="Test node 1",
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        input_schema={"input1": {"type": "string", "description": "Test input"}},
        output_schema={"output1": {"type": "string", "description": "Test output"}}
    )
    
    node2_config = NodeConfig(
        id="node2", 
        type="text_generation",
        model="gpt-4",
        prompt="test prompt 2",
        metadata=NodeMetadata(
            node_id="node2",
            node_type="text_generation",
            description="Test node 2",
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        input_schema={"input2": {"type": "string", "description": "Test input"}},
        output_schema={"output2": {"type": "string", "description": "Test output"}}
    )
    
    G.add_node("node1", config=node1_config)
    G.add_node("node2", config=node2_config)
    G.add_edge("node1", "node2")
    return G

@pytest.fixture
def context_manager(test_graph, vector_store):
    return GraphContextManager(
        graph=test_graph,
        max_tokens=1000,
        vector_store_config=vector_store.config.model_dump()
    )

@pytest.mark.asyncio
async def test_context_manager_initialization(context_manager, test_graph, vector_store):
    """Test that the context manager is initialized correctly."""
    assert context_manager.graph == test_graph
    assert context_manager.max_tokens == 1000
    
    # Test vector store configuration
    assert context_manager.vector_store.config.index_name == vector_store.config.index_name
    assert context_manager.vector_store.config.dimension == vector_store.config.dimension
    assert context_manager.vector_store.config.metric == vector_store.config.metric
    
    # Test context cache initialization
    assert isinstance(context_manager._context_cache, dict)
    assert len(context_manager._context_cache) == 0

@pytest.mark.asyncio
async def test_set_context(context_manager: GraphContextManager):
    """Test setting context for a node."""
    node_id = "node1"
    context = {
        "text": "This is a test context",
        "metadata": {
            "timestamp": "2024-03-20T12:00:00Z",
            "version": "1.0.0",
            "source": "test"
        },
        "embedding": [0.1] * 1536
    }
    
    await context_manager.set_context(node_id, context)
    stored_context = await context_manager.get_context(node_id)
    
    assert stored_context["text"] == context["text"]
    assert stored_context["metadata"] == context["metadata"]
    assert stored_context["embedding"] == context["embedding"]
    assert len(stored_context["embedding"]) == 1536

@pytest.mark.asyncio
async def test_get_optimized_context(context_manager: GraphContextManager):
    """Test getting optimized context with graph inheritance."""
    parent_context = {
        "text": "Parent context information",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    child_context = {
        "text": "Child context information",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    await context_manager.set_context("node1", parent_context)
    await context_manager.set_context("node2", child_context)
    
    optimized_context = await context_manager.get_context_with_optimization("node2")
    assert "text" in optimized_context
    assert optimized_context["text"] == child_context["text"]
    assert "metadata" in optimized_context
    assert "timestamp" in optimized_context["metadata"]
    assert "version" in optimized_context["metadata"]

@pytest.mark.asyncio
async def test_context_token_limits(context_manager: GraphContextManager):
    """Test that contexts are preserved when retrieved."""
    large_context = {
        "text": "Large " * 500,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    await context_manager.set_context("node1", large_context)
    retrieved_context = await context_manager.get_context_with_optimization("node1")
    
    assert "text" in retrieved_context
    assert retrieved_context["text"] == large_context["text"]
    assert "metadata" in retrieved_context
    assert "timestamp" in retrieved_context["metadata"]
    assert "version" in retrieved_context["metadata"]

@pytest.mark.asyncio
async def test_error_handling(context_manager):
    """Test error handling for invalid nodes and contexts."""
    # Test setting context for non-existent node
    with pytest.raises(ValueError, match="Node non_existent_node not found in graph"):
        await context_manager.set_context("non_existent_node", {
            "text": "Test context",
            "metadata": {"timestamp": datetime.now().isoformat(), "version": "1.0.0"}
        })
    
    # Test setting invalid context
    with pytest.raises(ValueError, match="Context must be a dictionary"):
        await context_manager.set_context("node1", "invalid_context")
    
    # Test getting context for non-existent node
    with pytest.raises(ValueError, match="Node non_existent_node not found in graph"):
        await context_manager.get_context("non_existent_node")

@pytest.mark.asyncio
async def test_context_cleanup(context_manager: GraphContextManager):
    """Test context cleanup functionality."""
    context1 = {
        "text": "First context",
        "metadata": {"timestamp": datetime.now().isoformat(), "version": "1.0.0"}
    }
    context2 = {
        "text": "Second context",
        "metadata": {"timestamp": datetime.now().isoformat(), "version": "1.0.0"}
    }
    
    await context_manager.set_context("node1", context1)
    await context_manager.set_context("node2", context2)
    
    await context_manager.clear_context("node1")
    assert await context_manager.get_context("node1") == {}
    assert await context_manager.get_context("node2") == context2
    
    await context_manager.clear_all_contexts()
    assert await context_manager.get_context("node1") == {}
    assert await context_manager.get_context("node2") == {}

@pytest.mark.asyncio
async def test_context_metadata(context_manager: GraphContextManager):
    """Test context metadata handling."""
    test_context = {
        "text": "Test context with metadata",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "custom_field": "custom_value"
        }
    }
    
    await context_manager.set_context("node1", test_context)
    context = await context_manager.get_context("node1")
    
    assert "metadata" in context
    assert "timestamp" in context["metadata"]
    assert "version" in context["metadata"]
    assert "custom_field" in context["metadata"]
    assert context["metadata"]["custom_field"] == "custom_value"

@pytest.mark.asyncio
async def test_graph_inheritance(context_manager: GraphContextManager):
    """Test context inheritance through graph structure."""
    parent_context = {
        "text": "Parent context",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    child_context = {
        "text": "Child context",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    
    await context_manager.set_context("node1", parent_context)
    await context_manager.set_context("node2", child_context)
    
    optimized = await context_manager.get_context_with_optimization("node2")
    
    assert "text" in optimized
    assert optimized["text"] == child_context["text"]
    assert "metadata" in optimized
    assert "timestamp" in optimized["metadata"]
    assert "version" in optimized["metadata"]

@pytest.mark.asyncio
async def test_pattern_extraction(context_manager):
    """Test pattern extraction from context."""
    test_input = {"input": "test input"}
    await context_manager.set_context("node1", test_input)
    pattern = await context_manager.extract_pattern("node1")
    assert pattern is not None
    assert "input_pattern" in pattern

@pytest.mark.asyncio
async def test_context_update_with_execution_result(context_manager):
    """Test updating context with NodeExecutionResult."""
    result = MockNodeExecutionResult(
        output="test output",
        success=True,
        node_id="node1"
    )
    await context_manager.update_context("node1", result)
    context = await context_manager.get_context("node1")
    assert context["output"]["text"] == "test output"
    assert context["success"] is True
    assert context["error"] is None
    assert "metadata" in context

@pytest.mark.asyncio
async def test_context_formatting(context_manager):
    """Test context formatting with different rules."""
    test_data = {
        "text": "plain text",
        "json": {"key": "value"},
        "code": "def test(): pass",
        "markdown": "# Title"
    }
    
    text_rule = ContextRule(include=True, format=ContextFormat.TEXT)
    formatted_text = await context_manager.format_context(test_data["text"], text_rule)
    assert formatted_text == "plain text"
    
    json_rule = ContextRule(include=True, format=ContextFormat.JSON)
    json_output = await context_manager.format_context(test_data["json"], json_rule)
    assert '"key": "value"' in json_output
    
    code_rule = ContextRule(include=True, format=ContextFormat.CODE)
    code_output = await context_manager.format_context(
        test_data["code"],
        code_rule,
        {"language": "python"}
    )
    assert "```python" in code_output
    assert "def test(): pass" in code_output

@pytest.mark.asyncio
async def test_vector_store_integration(context_manager, vector_store):
    """Test vector store integration with similarity search."""
    context1 = {
        "text": "This is the first context with specific information",
        "metadata": {"timestamp": datetime.now().isoformat(), "version": "1.0.0"}
    }
    context2 = {
        "text": "This is the second context with different information",
        "metadata": {"timestamp": datetime.now().isoformat(), "version": "1.0.0"}
    }
    
    await context_manager.set_context("node1", context1)
    await context_manager.set_context("node2", context2)
    
    results = await vector_store.search("test query", 2)
    assert len(results) == 2
    assert all(isinstance(r, VectorSearchResult) for r in results)
    assert all(r.score >= 0 and r.score <= 1 for r in results)

@pytest.mark.asyncio
async def test_context_cache_management(context_manager):
    """Test context cache management and cleanup."""
    # Add nodes to graph first
    for i in range(1200):
        node_id = f"node_{i}"
        node_config = NodeConfig(
            id=node_id,
            type="text_generation",
            model="gpt-4",
            prompt=f"test prompt {i}",
            metadata=NodeMetadata(
                node_id=node_id,
                node_type="text_generation",
                description=f"Test node {i}",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        )
        context_manager.graph.add_node(node_id, config=node_config)
        await context_manager.set_context(node_id, {"data": f"value_{i}"})
    
    # Verify cache size is maintained
    assert len(context_manager._context_cache) <= context_manager._max_cache_size
    
    # Verify oldest entries are removed
    assert "node_0" not in context_manager._context_cache
    assert f"node_{1199}" in context_manager._context_cache

@pytest.mark.asyncio
async def test_get_node_context_with_rules(context_manager):
    """Test getting node context with formatting rules."""
    await context_manager.set_context("node1", {
        "text": "plain text",
        "json": {"key": "value"},
        "code": "def test(): pass"
    })
    
    context_rules = {
        "text": ContextRule(include=True, format=ContextFormat.TEXT),
        "json": ContextRule(include=True, format=ContextFormat.JSON),
        "code": ContextRule(include=True, format=ContextFormat.CODE)
    }
    
    format_specs = {"code": {"language": "python"}}
    
    context = await context_manager.get_node_context(
        "node1",
        context_rules=context_rules,
        format_specs=format_specs
    )
    
    assert isinstance(context["text"], str)
    assert '"key": "value"' in context["json"]
    assert "```python" in context["code"] 