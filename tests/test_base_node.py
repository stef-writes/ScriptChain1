import pytest
from app.nodes.base import BaseNode
from app.models.node_models import NodeExecutionResult, NodeConfig, NodeMetadata

class TestNode(BaseNode):
    """Test implementation of BaseNode."""
    def __init__(self):
        config = NodeConfig(
            id="test_node",
            type="test_type",
            model="test_model",
            prompt="test prompt",
            metadata=NodeMetadata(
                node_id="test_node",
                node_type="test_type",
                version="1.0.0"
            )
        )
        super().__init__(config)
        
    async def execute(self, inputs=None):
        return NodeExecutionResult(
            output={"text": "test output"},
            success=True,
            error=None,
            metadata={
                "node_id": self.config.id,
                "node_type": self.config.type,
                "version": "1.0.0"
            }
        )

@pytest.fixture
def test_node():
    return TestNode()

@pytest.mark.asyncio
async def test_node_initialization(test_node):
    """Test node initialization."""
    assert test_node._execution_history == []
    assert await test_node.validate_input({}) is True

@pytest.mark.asyncio
async def test_execution_history(test_node):
    """Test execution history management."""
    # Execute node multiple times
    for i in range(15):  # More than max history size
        result = await test_node.execute()
        await test_node.post_execute(result)
    
    # Check history size is limited
    assert len(test_node._execution_history) == 10
    
    # Check history contains NodeExecutionResult objects
    for entry in test_node._execution_history:
        assert isinstance(entry, NodeExecutionResult)
        assert entry.output == {"text": "test output"}
        assert entry.success is True

@pytest.mark.asyncio
async def test_cleanup_state(test_node):
    """Test state cleanup functionality."""
    # Add some executions
    for i in range(5):
        result = await test_node.execute()
        await test_node.post_execute(result)
    
    initial_length = len(test_node._execution_history)
    await test_node.cleanup_state()
    assert len(test_node._execution_history) == initial_length

    # Add more executions to exceed limit
    for i in range(10):
        result = await test_node.execute()
        await test_node.post_execute(result)
    
    await test_node.cleanup_state()
    assert len(test_node._execution_history) == 10

@pytest.mark.asyncio
async def test_pre_execute_validation(test_node):
    """Test pre-execute input validation."""
    # Test with valid input
    await test_node.pre_execute({})
    
    # Test with invalid input
    async def mock_validate_input(context):
        return False
    test_node.validate_input = mock_validate_input
    with pytest.raises(ValueError):
        await test_node.pre_execute({}) 