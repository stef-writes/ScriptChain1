"""
Base node implementation defining the interface for all nodes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import ValidationError, create_model
from app.models.node_models import NodeConfig, NodeExecutionResult
import logging

logger = logging.getLogger(__name__)

class BaseNode(ABC):
    """Abstract base class for all nodes
    
    Provides:
    - Lifecycle hooks (pre_execute, post_execute)
    - Input validation using schema
    - Core node properties and configuration
    - State cleanup and execution history
    """
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self._execution_history = []  # Track last N executions
        self._max_history = 10  # Keep last 10 executions
        
    @property
    def node_id(self) -> str:
        """Get the node's unique identifier"""
        return self.config.metadata.node_id
    
    @property
    def node_type(self) -> str:
        """Get the node's type"""
        return self.config.metadata.node_type
    
    async def cleanup_state(self):
        """Clean up old execution records"""
        if len(self._execution_history) > self._max_history:
            self._execution_history = self._execution_history[-self._max_history:]
            logger.debug(f"Cleaned up execution history for node {self.node_id}")
    
    async def pre_execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and modify context before execution.
        
        This hook runs before execute() and can be used to:
        - Validate inputs
        - Transform context
        - Add additional context
        - Perform setup
        
        Args:
            context: The execution context
            
        Returns:
            Modified context dictionary
            
        Raises:
            ValueError: If context validation fails
        """
        # Validate inputs first
        if not await self.validate_input(context):
            raise ValueError(f"Input validation failed for node {self.node_id}")
            
        # Return the context (can be modified by subclasses)
        return context
        
    async def post_execute(self, result: NodeExecutionResult) -> NodeExecutionResult:
        """Process and potentially modify the execution result.
        
        This hook runs after execute() and can be used to:
        - Transform the result
        - Add additional metadata
        - Perform cleanup
        
        Args:
            result: The execution result
            
        Returns:
            Modified execution result
            
        Raises:
            ValueError: If result processing fails
        """
        # Add to execution history
        self._execution_history.append(result)
        await self.cleanup_state()
        
        # Return the result (can be modified by subclasses)
        return result
        
    async def validate_input(self, context: Dict[str, Any]) -> bool:
        """Validate the input context against the node's schema.
        
        Args:
            context: The execution context
            
        Returns:
            True if validation passes, False otherwise
        """
        # Default implementation always returns True
        # Subclasses should override this to implement validation
        return True
        
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> NodeExecutionResult:
        """Execute the node with the given context.
        
        Args:
            context: The execution context
            
        Returns:
            NodeExecutionResult containing the execution result
        """
        pass