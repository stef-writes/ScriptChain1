"""
API routes for the workflow engine
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from app.models.node_models import NodeConfig, NodeExecutionResult
from app.models.config import LLMConfig, MessageTemplate
from app.nodes.text_generation import TextGenerationNode
from app.chains.script_chain import ScriptChain
from app.utils.context import GraphContextManager
from app.models.vector_store import VectorSearchResult

router = APIRouter()

class NodeRequest(BaseModel):
    """Request model for node operations"""
    config: NodeConfig
    context: Optional[Dict[str, Any]] = None

class ChainRequest(BaseModel):
    """Request model for chain operations"""
    nodes: List[NodeConfig]
    context: Optional[Dict[str, Any]] = None

class TextSearchRequest(BaseModel):
    """Request model for text search"""
    query: str
    top_k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None
    namespace: str = "default"

class TextAddRequest(BaseModel):
    """Request model for adding texts"""
    texts: List[str]
    ids: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None
    namespace: str = "default"

@router.post("/nodes/text-generation", response_model=NodeExecutionResult)
async def create_text_generation_node(request: NodeRequest):
    """Create and execute a text generation node"""
    try:
        context_manager = GraphContextManager()
        node = TextGenerationNode(request.config, context_manager)
        result = await node.execute(request.context or {})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chains/execute", response_model=NodeExecutionResult)
async def execute_chain(request: ChainRequest):
    """Execute a chain of nodes"""
    try:
        context_manager = GraphContextManager()
        chain = ScriptChain(context_manager=context_manager)
        
        # Add nodes to chain
        for node_config in request.nodes:
            if node_config.metadata.node_type == "ai":
                node = TextGenerationNode(node_config, context_manager)
                chain.add_node(node)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported node type: {node_config.metadata.node_type}"
                )
        
        # Execute chain
        result = await chain.execute()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nodes/{node_id}/context")
async def get_node_context(node_id: str):
    """Get context for a specific node"""
    try:
        context_manager = GraphContextManager()
        context = context_manager.get_context(node_id)
        if context is None:
            raise HTTPException(status_code=404, detail=f"Context not found for node {node_id}")
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/nodes/{node_id}/context")
async def clear_node_context(node_id: str):
    """Clear context for a specific node"""
    try:
        context_manager = GraphContextManager()
        context_manager.set_context(node_id, {})  # Clear context by setting empty dict
        return {"message": f"Context cleared for node {node_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vectors/search", response_model=List[VectorSearchResult])
async def search_vectors(request: Request, search_request: TextSearchRequest):
    """Search for similar texts using vector similarity"""
    try:
        if not hasattr(request.app.state, "vector_store"):
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized"
            )
            
        results = await request.app.state.vector_store.search_text(
            query=search_request.query,
            top_k=search_request.top_k,
            filter_metadata=search_request.filter_metadata,
            namespace=search_request.namespace
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vectors/add")
async def add_texts(request: Request, add_request: TextAddRequest):
    """Add texts to the vector store"""
    try:
        if not hasattr(request.app.state, "vector_store"):
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized"
            )
            
        await request.app.state.vector_store.add_texts(
            texts=add_request.texts,
            ids=add_request.ids,
            metadata=add_request.metadata,
            namespace=add_request.namespace
        )
        
        return {"message": f"Added {len(add_request.texts)} texts to vector store"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vectors/stats")
async def get_vector_stats(request: Request):
    """Get statistics about the vector store"""
    try:
        if not hasattr(request.app.state, "vector_store"):
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized"
            )
            
        stats = await request.app.state.vector_store.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))