"""
Context management for LLM node execution
"""

from typing import Dict, List, Optional, Any, Union
import networkx as nx
from app.models.node_models import NodeExecutionResult, NodeMetadata, ContextFormat, ContextRule, InputMapping
from app.utils.logging import logger
from app.models.vector_store import VectorStoreConfig, SimilarityMetric
from app.vector.pinecone_store import PineconeVectorStore
from app.vector.pinecone_inference_store import PineconeInferenceVectorStore
import json
import os

class GraphContextManager:
    """Manages context for graph-based LLM node execution"""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        graph: Optional[nx.DiGraph] = None,
        vector_store_config: Optional[Dict] = None
    ):
        """Initialize context manager.
        
        Args:
            max_tokens: Maximum number of tokens to include in context
            graph: Optional graph structure for dependency tracking
            vector_store_config: Optional vector store configuration
        """
        self.max_tokens = max_tokens
        self.graph = graph or nx.DiGraph()
        
        # Initialize vector store if config provided
        if vector_store_config:
            vs_config = VectorStoreConfig(
                index_name=vector_store_config.get('index_name', 'default-index'),
                environment=vector_store_config.get('environment', 'us-west1'),
                dimension=vector_store_config.get('dimension', 1024),
                pod_type=vector_store_config.get('pod_type', 'p1'),
                replicas=vector_store_config.get('replicas', 1),
                use_inference=vector_store_config.get('use_inference', True),
                inference_model=vector_store_config.get('inference_model', 'llama-text-embed-v2'),
                api_key=vector_store_config.get('api_key', os.getenv('PINECONE_API_KEY')),
                host=vector_store_config.get('host'),
                metric=vector_store_config.get('metric', SimilarityMetric.COSINE)
            )
            
            # Use the appropriate vector store implementation
            if vs_config.use_inference:
                self.vector_store = PineconeInferenceVectorStore(vs_config)
            else:
                self.vector_store = PineconeVectorStore(vs_config)
        else:
            self.vector_store = None
            
        self.context_cache = {}
        self.format_handlers = {
            ContextFormat.TEXT: self._handle_text_format,
            ContextFormat.JSON: self._handle_json_format,
            ContextFormat.MARKDOWN: self._handle_markdown_format,
            ContextFormat.CODE: self._handle_code_format,
            ContextFormat.CUSTOM: self._handle_custom_format
        }
        
    def _handle_text_format(self, content: Any) -> str:
        """Handle text format conversion"""
        return str(content)

    def _handle_json_format(self, content: Any) -> str:
        """Handle JSON format conversion"""
        if isinstance(content, str):
            try:
                return json.dumps(json.loads(content), indent=2)
            except json.JSONDecodeError:
                return content
        return json.dumps(content, indent=2)

    def _handle_markdown_format(self, content: Any) -> str:
        """Handle markdown format conversion"""
        if isinstance(content, str):
            return content
        return f"```markdown\n{content}\n```"

    def _handle_code_format(self, content: Any) -> str:
        """Handle code format conversion"""
        if isinstance(content, str):
            return f"```\n{content}\n```"
        return f"```\n{str(content)}\n```"

    def _handle_custom_format(self, content: Any, format_spec: Dict[str, Any]) -> str:
        """Handle custom format conversion"""
        # Implement custom formatting logic based on format_spec
        return str(content)

    def format_context(self, content: Any, rule: ContextRule, format_specs: Optional[Dict[str, Any]] = None) -> str:
        """Format context according to specified rules"""
        if rule.format in self.format_handlers:
            if rule.format == ContextFormat.CUSTOM and format_specs:
                return self.format_handlers[rule.format](content, format_specs)
            return self.format_handlers[rule.format](content)
        return str(content)

    def get_node_context(
        self,
        node_id: str,
        selected_inputs: Optional[List[str]] = None,
        context_rules: Optional[Dict[str, ContextRule]] = None,
        format_specs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get context for a specific node.
        
        Args:
            node_id: ID of the node to get context for
            selected_inputs: Optional list of input keys to include
            context_rules: Optional rules for context formatting
            format_specs: Optional format specifications
            
        Returns:
            Dictionary containing the node's context
        """
        # Get the node's context
        context = self.get_context(node_id)
        
        # If no context rules provided, return the raw context
        if not context_rules:
            return context
            
        # Apply context rules
        formatted_context = {}
        for key, rule in context_rules.items():
            if key in context:
                formatted_context[key] = self.format_context(
                    context[key],
                    rule,
                    format_specs
                )
                
        return formatted_context
        
    def update_context(self, node_id: str, content: Any) -> None:
        """Update context for a node.
        
        Args:
            node_id: ID of the node to update context for
            content: Content to update context with
        """
        # If content is a NodeExecutionResult, extract the output
        if hasattr(content, 'output'):
            self.set_context(node_id, {'output': content.output})
        else:
            self.set_context(node_id, {'output': content})
            
        # Log the update
        logger.debug(f"Updated context for node {node_id}")
        
    def get_context(self, node_id: str) -> Dict[str, Any]:
        """Get the raw context for a node.
        
        Args:
            node_id: ID of the node to get context for
            
        Returns:
            Dictionary containing the node's context
        """
        # If the node has no context, return an empty dict
        if node_id not in self.context_cache:
            return {}
            
        return self.context_cache[node_id]
        
    def set_context(self, node_id: str, context: Dict[str, Any]) -> None:
        """Set the context for a node.
        
        Args:
            node_id: ID of the node to set context for
            context: Context to set
        """
        self.context_cache[node_id] = context
        logger.debug(f"Set context for node {node_id}")

    def get_node_output(self, node_id: str) -> Any:
        """Get the output of a specific node.
        
        Args:
            node_id: ID of the node to get output for
            
        Returns:
            The node's output from the context cache, or None if not found
            
        Raises:
            ValueError: If the node_id is not found in the graph
        """
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found in graph")
            
        # First check the context cache
        if node_id in self.context_cache:
            return self.context_cache[node_id]
            
        # If not in cache, return None and log warning
        logger.warning(f"No output found for node {node_id} in context cache")
        return None

    def clear_context(self, node_id: Optional[str] = None) -> None:
        """Clear context cache for a specific node or all nodes"""
        if node_id:
            self.context_cache.pop(node_id, None)
        else:
            self.context_cache.clear()

    def validate_context_rules(self, rules: Dict[str, ContextRule]) -> bool:
        """Validate context rules for consistency"""
        for node_id, rule in rules.items():
            if node_id not in self.graph.nodes:
                logger.warning(f"Context rule specified for non-existent node: {node_id}")
                return False
            if rule.max_tokens and rule.max_tokens > self.max_tokens:
                logger.warning(f"Context rule max_tokens exceeds system limit for node: {node_id}")
                return False
        return True

    async def get_context_with_optimization(
        self,
        node_id: str,
        query: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Get optimized context for a node using vector similarity.
        
        Args:
            node_id: ID of the node to get context for
            query: Query string for similarity search
            k: Number of similar contexts to retrieve
            threshold: Similarity threshold
            
        Returns:
            Dictionary containing optimized context
        """
        if not self.vector_store:
            return self.get_context(node_id)
            
        try:
            # Get base context
            base_context = self.get_context(node_id)
            
            # Get similar contexts from vector store
            if isinstance(self.vector_store, PineconeInferenceVectorStore):
                similar_contexts = await self.vector_store.search_text(
                    query=query,
                    top_k=k,
                    filter_metadata={"node_id": node_id}
                )
            else:
                # Fall back to the old method
                similar_contexts = await self.vector_store.similarity_search(
                    query,
                    k=k,
                    threshold=threshold
                )
            
            # Merge contexts while respecting token limit
            merged_context = self._merge_contexts(base_context, similar_contexts)
            return merged_context
            
        except Exception as e:
            logger.error(f"Error getting optimized context: {e}")
            return self.get_context(node_id)
            
    async def get_context_with_version(
        self,
        node_id: str,
        version: str = "latest"
    ) -> Dict[str, Any]:
        """Get versioned context for a node.
        
        Args:
            node_id: ID of the node to get context for
            version: Version of context to retrieve
            
        Returns:
            Dictionary containing versioned context
        """
        context = self.get_context(node_id)
        if version == "latest":
            return context
            
        # Get versioned context from vector store
        if self.vector_store:
            try:
                versioned_context = await self.vector_store.get_version(
                    node_id,
                    version
                )
                if versioned_context:
                    return versioned_context
            except Exception as e:
                logger.error(f"Error getting versioned context: {e}")
                
        return context
            
    def log_error(self, node_id: str, error: Exception) -> None:
        """Log error for a node.
        
        Args:
            node_id: ID of the node that encountered an error
            error: Exception that was raised
        """
        logger.error(f"Error in node {node_id}: {str(error)}")
        
    def _merge_contexts(
        self,
        base_context: Dict[str, Any],
        similar_contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge contexts while respecting token limit.
        
        Args:
            base_context: Base context dictionary
            similar_contexts: List of similar context dictionaries
            
        Returns:
            Merged context dictionary
        """
        # Implementation depends on specific token counting and merging logic
        # This is a placeholder that simply returns the base context
        return base_context