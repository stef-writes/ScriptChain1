"""
Context management for graph-based LLM node execution
"""

from typing import Dict, Any, List, Optional, Union
from app.models.node_models import NodeExecutionResult, ContextRule, ContextFormat
import networkx as nx
import logging
from datetime import datetime
import json
from app.models.vector_store import VectorStoreConfig
from app.context.vector import VectorStore

logger = logging.getLogger(__name__)

class GraphContextManager:
    """Manages context for graph-based LLM node execution"""
    
    def __init__(
        self,
        max_tokens: int = 1000,
        graph: Optional[nx.DiGraph] = None,
        vector_store_config: Optional[Union[Dict[str, Any], VectorStoreConfig]] = None
    ):
        """Initialize the context manager.
        
        Args:
            max_tokens: Maximum tokens allowed in context
            graph: Optional directed graph for context inheritance
            vector_store_config: Vector store configuration, can be a dictionary or VectorStoreConfig object
        """
        self.max_tokens = max_tokens
        self.graph = graph or nx.DiGraph()
        self._context_cache = {}
        self._learning_cache = []
        self._pattern_stats = {}
        self._max_cache_size = 1000
        self._max_learning_size = 100
        self.logger = logger
        
        if vector_store_config:
            # Handle both dict and VectorStoreConfig inputs
            config = (
                vector_store_config 
                if isinstance(vector_store_config, VectorStoreConfig)
                else VectorStoreConfig(**vector_store_config)
            )
            self.vector_store = VectorStore(
                index_name=config.index_name,
                dimension=config.dimension,
                metric=config.metric.value
            )
        else:
            self.vector_store = VectorStore(index_name="test_index")
            
    async def _cleanup_cache(self):
        """Clean up context cache if it exceeds maximum size."""
        if len(self._context_cache) > self._max_cache_size:
            # Remove oldest entries
            sorted_items = sorted(
                self._context_cache.items(),
                key=lambda x: x[1].get('timestamp', 0)
            )
            self._context_cache = dict(sorted_items[-self._max_cache_size:])
            self.logger.debug("Cleaned up context cache")
            
    async def _cleanup_learning_cache(self):
        """Remove oldest entries from learning cache"""
        if len(self._learning_cache) > self._max_learning_size:
            # Remove least successful patterns
            sorted_patterns = sorted(
                self._pattern_stats.items(),
                key=lambda x: x[1]['success_rate']
            )
            remove_count = int(self._max_learning_size * 0.2)
            for node_id, _ in sorted_patterns[:remove_count]:
                self._learning_cache.pop(node_id, None)
                self._pattern_stats.pop(node_id, None)
            logger.debug("Cleaned up learning cache")
            
    async def _extract_pattern(self, result: NodeExecutionResult) -> Optional[Dict]:
        """Extract reusable patterns from successful executions"""
        if not hasattr(result, 'output') or not result.output:
            return None
        
        # Extract key patterns with metadata
        pattern = {
            'input_pattern': getattr(result, 'context_used', {}),
            'output_pattern': result.output,
            'success_metrics': getattr(result, 'usage', {}),
            'timestamp': datetime.now().isoformat(),
            'node_type': getattr(result, 'node_type', 'unknown')
        }
        
        # Update pattern statistics
        node_id = getattr(result, 'node_id', 'unknown')
        if node_id not in self._pattern_stats:
            self._pattern_stats[node_id] = {
                'success_count': 0,
                'total_count': 0,
                'success_rate': 0.0
            }
        
        stats = self._pattern_stats[node_id]
        stats['total_count'] += 1
        if getattr(result, 'success', False):
            stats['success_count'] += 1
        stats['success_rate'] = stats['success_count'] / stats['total_count']
        
        return pattern
        
    async def get_node_context(
        self,
        node_id: str,
        context_rules: Optional[Dict[str, ContextRule]] = None,
        format_specs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get context for a node with optional formatting rules."""
        context = await self.get_context(node_id) or {}
        
        if context_rules:
            formatted_context = {}
            for key, value in context.items():
                if key in context_rules:
                    rule = context_rules[key]
                    if rule.include:
                        formatted_context[key] = await self.format_context(
                            value,
                            rule,
                            format_specs.get(key) if format_specs else None
                        )
            return formatted_context
        return context
        
    async def format_context(
        self,
        value: Any,
        rule: ContextRule,
        format_specs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format context value according to rule."""
        if rule.format == ContextFormat.TEXT:
            return str(value)
        elif rule.format == ContextFormat.JSON:
            return json.dumps(value, indent=2)
        elif rule.format == ContextFormat.CODE:
            lang = format_specs.get('language', '') if format_specs else ''
            return f"```{lang}\n{value}\n```"
        elif rule.format == ContextFormat.MARKDOWN:
            return f"```\n{value}\n```"
        return str(value)

    async def get_context(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get raw context for a node."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found in graph")
        return self._context_cache.get(node_id, {})
        
    async def set_context(self, node_id: str, context: Dict[str, Any]):
        """Set context for a node."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found in graph")
        if not isinstance(context, dict):
            raise ValueError("Context must be a dictionary")
        self._context_cache[node_id] = context
        self.logger.debug(f"Set context for node {node_id}")
        await self._cleanup_cache()
        
    async def update_context(self, node_id: str, context: Union[Dict[str, Any], NodeExecutionResult]):
        """Update context for a node."""
        if isinstance(context, NodeExecutionResult):
            current = await self.get_context(node_id) or {}
            current.update({
                'output': context.output,
                'success': context.success,
                'error': context.error,
                'metadata': context.metadata,
                'timestamp': context.metadata.timestamp
            })
            self._context_cache[node_id] = current
        else:
            current = await self.get_context(node_id) or {}
            current.update(context)
            self._context_cache[node_id] = current
        
        self.logger.debug(f"Updated context for node {node_id}")
        await self._cleanup_cache()

    async def clear_context(self, node_id: str):
        """Clear context for a specific node."""
        if node_id in self._context_cache:
            del self._context_cache[node_id]
            self.logger.debug(f"Cleared context for node {node_id}")

    async def clear_all_contexts(self):
        """Clear all contexts."""
        self._context_cache.clear()
        self.logger.debug("Cleared all contexts")

    async def get_context_with_optimization(self, node_id: str) -> Dict[str, Any]:
        """Get optimized context with graph inheritance."""
        context = await self.get_context(node_id) or {}
        
        # Get parent contexts
        for parent in self.graph.predecessors(node_id):
            parent_context = await self.get_context(parent) or {}
            # Don't override child context with parent
            for k, v in parent_context.items():
                if k not in context:
                    context[k] = v
        
        return context

    async def get_context_with_version(self, node_id: str) -> Dict[str, Any]:
        """Get context with version metadata."""
        context = await self.get_context(node_id) or {}
        return {
            'version': '1.0',
            'timestamp': context.get('timestamp', None),
            'data': context
        }

    async def extract_pattern(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Extract pattern from node context."""
        context = await self.get_context(node_id)
        if not context:
            return {}
            
        return {
            'input_pattern': context,
            'timestamp': context.get('timestamp', None)
        }