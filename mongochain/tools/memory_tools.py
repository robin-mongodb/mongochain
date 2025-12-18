"""Memory-related tools for agents."""

from typing import Optional, Dict, Any, List
from mongochain.core.schemas import SearchResult


class MemoryStoreTool:
    """Tool for agents to store memories."""
    
    def __init__(self, agent):
        """
        Initialize MemoryStoreTool.
        
        Args:
            agent: MongoAgent instance
        """
        self.agent = agent
        self.name = "store_memory"
        self.description = "Store important information as a memory for future reference"
    
    def __call__(
        self,
        content: str,
        memory_type: str = "semantic",
        namespace: str = "default",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a memory.
        
        Args:
            content: What to remember
            memory_type: Type of memory (semantic, episodic, procedural)
            namespace: User or session identifier
            importance: How important this memory is (0-1)
            metadata: Additional context
        
        Returns:
            Result dict with memory_id and status
        """
        try:
            memory_id = self.agent.store_memory(
                content=content,
                memory_type=memory_type,
                namespace=namespace,
                metadata=metadata or {},
                importance=importance
            )
            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Memory stored successfully: {content[:50]}..."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to store memory"
            }


class MemorySearchTool:
    """Tool for agents to search memories."""
    
    def __init__(self, agent):
        """
        Initialize MemorySearchTool.
        
        Args:
            agent: MongoAgent instance
        """
        self.agent = agent
        self.name = "search_memory"
        self.description = "Search for relevant past memories or conversations"
    
    def __call__(
        self,
        query: str,
        namespace: str = "default",
        memory_type: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search memories.
        
        Args:
            query: What to search for
            namespace: User or session identifier
            memory_type: Filter by memory type (optional)
            top_k: Number of results
        
        Returns:
            Result dict with search results
        """
        try:
            results = self.agent.search_memory(
                query=query,
                namespace=namespace,
                memory_type=memory_type,
                top_k=top_k
            )
            
            formatted_results = [
                {
                    "content": result.memory.content,
                    "type": result.memory.memory_type,
                    "importance": result.memory.importance,
                    "score": result.score
                }
                for result in results
            ]
            
            return {
                "success": True,
                "results": formatted_results,
                "count": len(formatted_results),
                "message": f"Found {len(formatted_results)} relevant memories"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to search memories"
            }


class MemoryUpdateTool:
    """Tool for agents to update existing memories."""
    
    def __init__(self, agent):
        """
        Initialize MemoryUpdateTool.
        
        Args:
            agent: MongoAgent instance
        """
        self.agent = agent
        self.name = "update_memory"
        self.description = "Update an existing memory with new information"
    
    def __call__(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update a memory (placeholder - will implement in memory_store).
        
        Args:
            memory_id: ID of memory to update
            new_content: Updated content
            metadata: Updated metadata
        
        Returns:
            Result dict with status
        """
        return {
            "success": False,
            "message": "Update memory feature coming soon",
            "placeholder": True
        }
