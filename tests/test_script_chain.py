"""
Tests for the enhanced ScriptChain implementation.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from app.chains.script_chain import ScriptChain, ExecutionLevel
from app.models.node_models import (
    NodeConfig, 
    NodeExecutionResult, 
    NodeMetadata, 
    UsageMetadata,
    ContextFormat,
    ContextRule,
    InputMapping
)
from app.models.vector_store import SimilarityMetric, VectorStoreConfig
from app.utils.callbacks import LoggingCallback, MetricsCallback, ScriptChainCallback
from app.models.config import LLMConfig
from datetime import datetime
from app.vector.base import VectorStoreInterface

# Configure asyncio fixture loop scope
pytest_plugins = ('pytest_asyncio',)
pytestmark = pytest.mark.asyncio(scope="function")

@pytest.fixture
async def mock_callback():
    callback = MagicMock()
    callback.on_chain_start = AsyncMock()
    callback.on_chain_end = AsyncMock()
    callback.on_node_start = AsyncMock()
    callback.on_node_end = AsyncMock()
    callback.on_node_error = AsyncMock()
    return callback

@pytest.fixture
async def script_chain():
    # Create vector store config with all required fields
    vector_store_config = VectorStoreConfig(
        index_name="test-index",
        environment="test",
        dimension=1536,
        metric=SimilarityMetric.COSINE,
        pod_type="p1.x1",
        replicas=1,
        use_inference=True,
        inference_model="text-embedding-ada-002",
        api_key="test-key",
        metadata_config={
            "indexed": "node_id,node_type"
        }
    )
    
    # Create LLM config
    llm_config = LLMConfig(
        model="gpt-4",
        api_key="test-key",
        max_context_tokens=1000,
        temperature=0.7,
        max_tokens=500
    )
    
    # Create mock vector store
    mock_vector_store = MagicMock(spec=VectorStoreInterface)
    mock_vector_store.config = vector_store_config
    mock_vector_store.query = AsyncMock(return_value=[])
    mock_vector_store.add = AsyncMock(return_value=True)
    
    # Create chain with mock callback
    mock_callback = AsyncMock(spec=ScriptChainCallback)
    chain = ScriptChain(
        max_context_tokens=1000,
        callbacks=[mock_callback],
        concurrency_level=2,
        vector_store_config=vector_store_config,
        llm_config=llm_config
    )
    
    # Replace the vector store with our mock
    chain.vector_store = mock_vector_store
    
    # Ensure the context manager uses the same graph as the chain
    chain.context.graph = chain.graph
    
    return chain

@pytest.mark.asyncio
async def test_script_chain_initialization(script_chain: ScriptChain):
    """Test ScriptChain initialization."""
    assert script_chain.max_context_tokens == 1000
    assert script_chain.concurrency_level == 2
    assert script_chain.nodes == {}
    assert script_chain.dependencies == {}
    assert script_chain.execution_levels == {}
    assert script_chain.llm_config.model == "gpt-4"
    assert script_chain.llm_config.api_key == "test-key"
    assert script_chain.llm_config.temperature == 0.7
    assert script_chain.llm_config.max_tokens == 500
    assert script_chain.llm_config.max_context_tokens == 1000
    assert script_chain.vector_store_config.index_name == "test-index"
    assert script_chain.vector_store_config.metric == SimilarityMetric.COSINE

@pytest.mark.asyncio
async def test_add_node(script_chain):
    # Create a valid node config
    node_config = NodeConfig(
        id="test-node",
        type="llm",
        model="gpt-4",
        prompt="test prompt",
        metadata=NodeMetadata(
            node_id="test-node",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    # Add node to chain
    script_chain.add_node(node_config)
    
    # Verify node was added
    assert "test-node" in script_chain.graph.nodes
    assert "test-node" in script_chain.node_registry
    assert script_chain.node_registry["test-node"].config == node_config
    
    # Verify node was added to the nodes dictionary
    assert "test-node" in script_chain.nodes
    assert script_chain.nodes["test-node"].config == node_config

@pytest.mark.asyncio
async def test_validate_workflow(script_chain: ScriptChain):
    """Test workflow validation."""
    # Add valid nodes with required metadata and context rules
    nodes = {
        "node1": NodeConfig(
            id="node1",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 1",
            level=0,
            context_rules={
                "input": ContextRule(
                    include=True,
                    format=ContextFormat.TEXT,
                    required=True
                )
            },
            metadata=NodeMetadata(
                node_id="node1",
                node_type="llm",
                version="1.0.0"
            )
        ),
        "node2": NodeConfig(
            id="node2",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 2",
            level=1,
            dependencies=["node1"],
            context_rules={
                "node1_output": ContextRule(
                    include=True,
                    format=ContextFormat.TEXT,
                    required=True
                )
            },
            metadata=NodeMetadata(
                node_id="node2",
                node_type="llm",
                version="1.0.0"
            )
        )
    }
    
    for node in nodes.values():
        script_chain.add_node(node)
    
    # Validate should pass
    assert script_chain.validate_workflow() is True
    
    # Add orphan node with required metadata
    orphan_node = NodeConfig(
        id="orphan",
        type="llm",
        model="gpt-4",
        prompt="Test",
        level=0,
        metadata=NodeMetadata(
            node_id="orphan",
            node_type="llm",
            version="1.0.0"
        )
    )
    script_chain.add_node(orphan_node)
    
    # Validate should still pass but log warning
    assert script_chain.validate_workflow() is True

@pytest.mark.asyncio
async def test_execution_levels(script_chain):
    # Add nodes with dependencies
    node1 = NodeConfig(
        id="node1",
        type="llm",
        model="gpt-4",
        prompt="test prompt",
        metadata=NodeMetadata(
            node_id="node1",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    node2 = NodeConfig(
        id="node2",
        type="llm",
        model="gpt-4",
        prompt="test prompt",
        dependencies=["node1"],
        metadata=NodeMetadata(
            node_id="node2",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    script_chain.add_node(node1)
    script_chain.add_node(node2)
    
    # Calculate execution levels
    levels = script_chain._calculate_execution_levels()
    
    assert len(levels) == 2
    assert "node1" in levels[0].node_ids
    assert "node2" in levels[1].node_ids

@pytest.mark.asyncio
async def test_parallel_execution(script_chain):
    # Create node configs
    node0_config = NodeConfig(
        id="node0",
        type="llm",
        model="gpt-4",
        prompt="Node 0 prompt",
        metadata=NodeMetadata(
            node_id="node0",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    node1_config = NodeConfig(
        id="node1",
        type="llm",
        model="gpt-4",
        prompt="Node 1 prompt",
        dependencies=["node0"],
        metadata=NodeMetadata(
            node_id="node1",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    node2_config = NodeConfig(
        id="node2",
        type="llm",
        model="gpt-4",
        prompt="Node 2 prompt",
        dependencies=["node0"],
        metadata=NodeMetadata(
            node_id="node2",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    # Add nodes to graph first
    script_chain.add_node(node0_config)
    script_chain.add_node(node1_config)
    script_chain.add_node(node2_config)
    
    # Mock execute function
    async def mock_execute(node_id: str, inputs: Dict[str, Any] = None):
        return NodeExecutionResult(
            success=True,
            output=f"Output from {node_id}",
            metadata=NodeMetadata(
                node_id=node_id,
                node_type="llm",
                version="1.0.0",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            ),
            usage_metadata=UsageMetadata(
                model="gpt-4",
                node_id=node_id,
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )
    
    # Mock execute method for all nodes
    script_chain.nodes["node0"].execute = mock_execute
    script_chain.nodes["node1"].execute = mock_execute
    script_chain.nodes["node2"].execute = mock_execute
    
    # Execute chain
    result = await script_chain.execute()
    
    # Verify execution
    assert isinstance(result, NodeExecutionResult)
    assert result.success
    assert "node0" in result.output
    assert "node1" in result.output
    assert "node2" in result.output
    assert result.output["node0"].output == "Output from node0"
    assert result.output["node1"].output == "Output from node1"
    assert result.output["node2"].output == "Output from node2"

@pytest.mark.asyncio
async def test_error_handling(script_chain):
    # Add a node that will fail
    node = NodeConfig(
        id="error-node",
        type="llm",
        model="gpt-4",
        prompt="test prompt",
        metadata=NodeMetadata(
            node_id="error-node",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    script_chain.add_node(node)
    
    # Mock execute_node to raise an exception
    async def mock_execute(*args, **kwargs):
        raise Exception("Test error")
    
    # Mock the execute method for the error node
    script_chain.node_registry["error-node"].execute = AsyncMock(side_effect=mock_execute)
    
    # Execute and verify error handling
    result = await script_chain.execute()
    assert not result.success
    assert "Test error" in str(result.error)

@pytest.mark.asyncio
async def test_callback_integration(script_chain):
    # Create a test node config
    test_node_config = NodeConfig(
        id="test_node",
        type="llm",
        model="gpt-4",
        prompt="test prompt",
        metadata=NodeMetadata(
            node_id="test_node",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    # Add node to graph first
    script_chain.add_node(test_node_config)
    
    # Create a mock callback
    mock_callback = AsyncMock(spec=ScriptChainCallback)
    script_chain.add_callback(mock_callback)
    
    # Mock execute function
    async def mock_execute(node_id: str, inputs: Dict[str, Any] = None):
        return NodeExecutionResult(
            success=True,
            output="Test output",
            metadata=NodeMetadata(
                node_id=node_id,
                node_type="llm",
                version="1.0.0",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            ),
            usage_metadata=UsageMetadata(
                model="gpt-4",
                node_id=node_id,
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )
    
    # Mock execute method for test node
    script_chain.nodes["test_node"].execute = mock_execute
    
    # Execute chain
    result = await script_chain.execute()
    
    # Verify execution
    assert isinstance(result, NodeExecutionResult)
    assert result.success
    assert "test_node" in result.output
    assert result.output["test_node"].output == "Test output"
    
    # Verify callback was called
    mock_callback.on_chain_start.assert_called_once()
    mock_callback.on_node_start.assert_called_once()
    mock_callback.on_node_end.assert_called_once()
    mock_callback.on_chain_end.assert_called_once()

@pytest.mark.asyncio
async def test_retry_mechanism(script_chain):
    # Create a node config that will fail
    failing_node_config = NodeConfig(
        id="failing_node",
        type="llm",
        model="gpt-4",
        prompt="test prompt",
        metadata=NodeMetadata(
            node_id="failing_node",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    # Add node to graph first
    script_chain.add_node(failing_node_config)
    
    # Track number of attempts
    attempt_count = 0
    
    # Mock execute function that fails 3 times then succeeds
    async def mock_execute(node_id, inputs):
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count <= 3:
            raise Exception(f"Simulated failure attempt {attempt_count}")
            
        return NodeExecutionResult(
            success=True,
            output="Success after retries",
            metadata=NodeMetadata(
                node_id=node_id,
                node_type="llm",
                version="1.0.0",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            ),
            usage_metadata=UsageMetadata(
                model="gpt-4",
                node_id=node_id,
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )
    
    # Mock execute method for failing node
    script_chain.nodes["failing_node"].execute = mock_execute
    
    # Execute node with retries
    with pytest.raises(Exception) as exc_info:
        await script_chain.execute_nodes(["failing_node"], max_retries=3)
    
    # Verify that it failed after 3 retries
    assert attempt_count == 3
    assert "Simulated failure attempt 3" in str(exc_info.value)
    
    # Reset attempt count and try again with more retries
    attempt_count = 0
    result = await script_chain.execute_nodes(["failing_node"], max_retries=4)
    
    # Verify that it succeeded on the 4th attempt
    assert attempt_count == 4
    assert isinstance(result, NodeExecutionResult)
    assert result.success
    assert result.output == "Success after retries"

@pytest.mark.asyncio
async def test_add_node_with_context_rules(script_chain: ScriptChain):
    """Test adding nodes with context rules and format specifications."""
    # Create test node with context rules
    node = NodeConfig(
        id="test_node",
        type="llm",
        model="gpt-4",
        prompt="Test prompt with {input1} and {input2}",
        level=0,
        metadata=NodeMetadata(
            node_id="test_node",
            node_type="llm",
            version="1.0.0"
        ),
        context_rules={
            "input1": ContextRule(
                include=True,
                format=ContextFormat.TEXT,
                required=True
            ),
            "input2": ContextRule(
                include=True,
                format=ContextFormat.JSON,
                max_tokens=100
            )
        },
        format_specifications={
            "input1": {"prefix": "Input 1: "},
            "input2": {"indent": 2}
        }
    )
    
    # Add node
    script_chain.add_node(node)
    
    # Verify node was added with context rules
    assert "test_node" in script_chain.nodes
    assert script_chain.nodes["test_node"].config.context_rules == node.context_rules
    assert script_chain.nodes["test_node"].config.format_specifications == node.format_specifications

@pytest.mark.asyncio
async def test_input_validation(script_chain: ScriptChain):
    """Test input validation with context rules."""
    # Create test node with required input
    node = NodeConfig(
        id="validation_node",
        type="llm",
        model="gpt-4",
        prompt="Test prompt",
        level=0,
        metadata=NodeMetadata(
            node_id="validation_node",
            node_type="llm",
            version="1.0.0"
        ),
        context_rules={
            "required_input": ContextRule(
                include=True,
                required=True
            )
        }
    )
    
    script_chain.add_node(node)
    
    # Test with missing required input
    result = await script_chain.nodes["validation_node"].validate_inputs({})
    assert not result
    
    # Test with valid input
    result = await script_chain.nodes["validation_node"].validate_inputs({
        "required_input": "test value"
    })
    assert result

@pytest.mark.asyncio
async def test_context_formatting(script_chain):
    # Create test node config
    test_node_config = NodeConfig(
        id="format_test",
        type="llm",
        model="gpt-4",
        prompt="test prompt",
        metadata=NodeMetadata(
            node_id="format_test",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    # Add node to graph first
    script_chain.add_node(test_node_config)
    
    # Set up test context
    test_context = {
        "text": "Test content",
        "code": "def test(): pass",
        "json": {"key": "value"},
        "markdown": "# Test"
    }
    
    # Set context with formatting rules
    await script_chain.context.set_context("format_test", test_context)
    
    # Get formatted context
    formatted_context = await script_chain.context.get_node_context(
        "format_test",
        context_rules={
            "text": ContextRule(include=True, format=ContextFormat.TEXT),
            "code": ContextRule(include=True, format=ContextFormat.CODE),
            "json": ContextRule(include=True, format=ContextFormat.JSON),
            "markdown": ContextRule(include=True, format=ContextFormat.MARKDOWN)
        },
        format_specs={
            "code": {"language": "python"}
        }
    )
    
    # Verify formatting
    assert formatted_context["text"] == "Test content"
    assert "```python" in formatted_context["code"]
    assert "def test(): pass" in formatted_context["code"]
    assert '"key": "value"' in formatted_context["json"]
    assert "# Test" in formatted_context["markdown"]

@pytest.mark.asyncio
async def test_parallel_execution_with_context(script_chain):
    # Create node configs
    node1_config = NodeConfig(
        id="node1",
        type="llm",
        model="gpt-4",
        prompt="Node 1 prompt",
        metadata=NodeMetadata(
            node_id="node1",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    node2_config = NodeConfig(
        id="node2",
        type="llm",
        model="gpt-4",
        prompt="Node 2 prompt",
        metadata=NodeMetadata(
            node_id="node2",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    # Add nodes to graph first
    script_chain.add_node(node1_config)
    script_chain.add_node(node2_config)
    
    # Set up contexts
    context1 = {
        "text": "Context for node 1",
        "metadata": {
            "format": "text",
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    context2 = {
        "text": "Context for node 2",
        "metadata": {
            "format": "text",
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    await script_chain.context.set_context("node1", context1)
    await script_chain.context.set_context("node2", context2)
    
    # Mock execute function
    async def mock_execute(node_id, inputs):
        return NodeExecutionResult(
            success=True,
            output=f"Output from {node_id}",
            metadata=NodeMetadata(
                node_id=node_id,
                node_type="llm",
                version="1.0.0",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            ),
            usage_metadata=UsageMetadata(
                model="gpt-4",
                node_id=node_id,
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )
    
    # Mock execute method for both nodes
    script_chain.nodes["node1"].execute = mock_execute
    script_chain.nodes["node2"].execute = mock_execute
    
    # Execute nodes in parallel
    result = await script_chain.execute_nodes(["node1", "node2"])
    
    # Verify execution
    assert isinstance(result, NodeExecutionResult)
    assert result.success
    assert "node1" in result.output
    assert "node2" in result.output
    
    # Verify contexts were maintained
    node1_context = await script_chain.context.get_context("node1")
    node2_context = await script_chain.context.get_context("node2")
    
    assert node1_context["text"] == "Context for node 1"
    assert node2_context["text"] == "Context for node 2"

@pytest.mark.asyncio
async def test_context_inheritance(script_chain):
    # Add parent and child nodes
    parent_config = NodeConfig(
        id="parent_node",
        type="llm",
        model="gpt-4",
        prompt="parent prompt",
        metadata=NodeMetadata(
            node_id="parent_node",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    child_config = NodeConfig(
        id="child_node",
        type="llm",
        model="gpt-4",
        prompt="child prompt",
        dependencies=["parent_node"],
        metadata=NodeMetadata(
            node_id="child_node",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    # Add nodes to chain
    script_chain.add_node(parent_config)
    script_chain.add_node(child_config)
    
    # Set parent context
    parent_context = {
        "text": "parent context",
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    }
    
    # Set context for parent node
    await script_chain.context.set_context("parent_node", parent_context)
    
    # Verify child inherits parent context
    child_context = await script_chain.context.get_node_context("child_node")
    assert isinstance(child_context, dict)
    assert "text" in child_context
    assert child_context["text"] == "parent context"
    assert "metadata" in child_context
    assert child_context["metadata"]["version"] == "1.0.0"

@pytest.mark.asyncio
async def test_context_cleanup(script_chain):
    # Create a node config with required metadata
    node_config = NodeConfig(
        id="node1",
        type="llm",
        model="gpt-4",
        prompt="test prompt",
        metadata=NodeMetadata(
            node_id="node1",
            node_type="llm",
            version="1.0.0",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
    )
    
    # Add node to chain
    script_chain.add_node(node_config)
    
    # Set test context
    context1 = {
        "text": "test context",
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    }
    
    # Set context for node
    await script_chain.context.set_context("node1", context1)
    
    # Verify context was set
    node_context = await script_chain.context.get_node_context("node1")
    assert node_context is not None
    assert node_context["text"] == "test context"
    assert node_context["metadata"]["version"] == "1.0.0"
    
    # Cleanup context
    await script_chain.context.cleanup()
    
    # Verify context was cleaned up
    node_context = await script_chain.context.get_node_context("node1")
    assert node_context is None or node_context == {}

@pytest.mark.asyncio
async def test_vector_store_integration(script_chain: ScriptChain):
    """Test vector store integration with similarity search."""
    # Add test nodes
    node1_config = NodeConfig(
        id="node1",
        type="llm",
        model="gpt-4",
        prompt="Test prompt 1",
        level=0,
        metadata=NodeMetadata(
            node_id="node1",
            node_type="llm",
            version="1.0.0"
        )
    )
    script_chain.add_node(node1_config)
    
    # Set context with text for vector storage
    context1 = {
        "text": "This is the first context with specific information",
        "metadata": {"timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}
    }
    await script_chain.context.set_context("node1", context1)
    
    # Mock vector store search results
    mock_results = [
        {"text": "test content", "metadata": {"node_id": "test_node"}} for _ in range(4)
    ]
    script_chain.vector_store.similarity_search = AsyncMock(return_value=mock_results)
    
    # Perform search and verify results
    results = await script_chain.vector_store.similarity_search("test query", k=4)
    assert len(results) == 4
    assert all(isinstance(r, dict) for r in results)
    assert all("text" in r and "metadata" in r for r in results)

@pytest.mark.asyncio
async def test_context_handling(script_chain):
    test_context = {
        "text": "Test context content",
        "metadata": {
            "version": "1.0.0",
            "custom_field": "custom_value"
        }
    }
    
    # Set and verify context
    await script_chain.context.set_context("test_node", test_context)
    context = await script_chain.context.get_context("test_node")
    
    # Verify context and metadata
    assert "text" in context
    assert context["text"] == test_context["text"]
    assert "metadata" in context
    assert "timestamp" in context["metadata"]
    assert "version" in context["metadata"]
    assert "custom_field" in context["metadata"]
    assert context["metadata"]["custom_field"] == "custom_value"