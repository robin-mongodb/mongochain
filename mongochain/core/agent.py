"""MongoAgent - Agent with MongoDB memory backend."""

from typing import Optional, Any, Dict, List
from mongochain.core.schemas import AgentConfig
from mongochain.core.memory_store import MongoMemoryStore
from mongochain.core.config import MongoChainConfig


class MongoAgent:
    """
    AI Agent with MongoDB-backed memory.
    
    Each agent gets its own MongoDB database named after the agent.
    The connection string must be set once globally using set_connection_string().
    """
    
    def __init__(
        self,
        name: str,
        model: str = "gpt-4",
        embedding_model: str = "text-embedding-3-small",
        persona: Optional[str] = None,
        collaborator_agents: Optional[List[str]] = None,
        enable_short_term: bool = True,
        enable_long_term: bool = True,
        short_term_ttl_days: int = 90,
        max_context_tokens: int = 4000,
        **kwargs
    ):
        """
        Initialize MongoAgent.
        
        Args:
            name: Agent name (used as database name)
            model: LLM model name (e.g., "gpt-4", "claude-3-sonnet")
            embedding_model: Embedding model name (e.g., "text-embedding-3-small", "voyage-3")
            persona: Agent persona/personality description
            collaborator_agents: List of agent names this agent can collaborate with
            enable_short_term: Enable short-term memory
            enable_long_term: Enable long-term memory (user profiles)
            short_term_ttl_days: Days before short-term memories expire (default 90 days)
            max_context_tokens: Maximum tokens for context window
        """
        # Get connection string from global config
        connection_string = MongoChainConfig.get_connection_string()
        
        # Validate agent name
        self.name = self._validate_agent_name(name)
        
        # Create agent configuration
        self.config = AgentConfig(
            name=self.name,
            connection_string=connection_string,
            model=model,
            embedding_model=embedding_model,
            persona=persona,
            collaborator_agents=collaborator_agents or [],
            enable_short_term=enable_short_term,
            enable_long_term=enable_long_term,
            short_term_ttl_days=short_term_ttl_days,
            max_context_tokens=max_context_tokens
        )
        
        # Initialize memory store with agent-specific database
        self.memory_store = MongoMemoryStore(
            connection_string=connection_string,
            database_name=self.name,
            embedding_model=embedding_model,
            short_term_ttl_days=short_term_ttl_days
        )
        
        print(f"✓ MongoAgent '{self.name}' initialized")
        print(f"  Database: {self.name}")
        print(f"  Model: {model}")
        if persona:
            print(f"  Persona: {persona}")
        if collaborator_agents:
            print(f"  Collaborators: {', '.join(collaborator_agents)}")
    
    def _validate_agent_name(self, name: str) -> str:
        """
        Validate and sanitize agent name for use as MongoDB database name.
        
        MongoDB database names must:
        - Not be empty
        - Not contain: /\\. "$*<>:|?
        - Be less than 64 characters
        - Not start with 'system.'
        """
        if not name:
            raise ValueError("Agent name cannot be empty")
        
        # Replace invalid characters with underscores
        invalid_chars = '/\\. "$*<>:|?'
        sanitized = name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Check length
        if len(sanitized) > 63:
            sanitized = sanitized[:63]
        
        # Check system prefix
        if sanitized.lower().startswith('system.'):
            raise ValueError("Agent name cannot start with 'system.'")
        
        return sanitized
    
    def store_memory(
        self,
        content: str,
        memory_type: str = "semantic",
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> str:
        """
        Store a new memory.
        
        Args:
            content: Memory content to store
            memory_type: Type of memory ("semantic", "episodic", "procedural")
            namespace: User or session identifier
            metadata: Additional metadata
            importance: Memory importance (0-1)
        
        Returns:
            Memory ID
        """
        return self.memory_store.store_memory(
            content=content,
            memory_type=memory_type,
            namespace=namespace,
            metadata=metadata or {},
            importance=importance
        )
    
    def search_memory(
        self,
        query: str,
        memory_type: Optional[str] = None,
        namespace: str = "default",
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ):
        """
        Search memories using semantic similarity.
        
        Args:
            query: Search query
            memory_type: Filter by memory type (optional)
            namespace: User or session identifier
            top_k: Number of results to return
            filters: Additional MongoDB filters
        
        Returns:
            List of SearchResult objects
        """
        return self.memory_store.search_memory(
            query=query,
            memory_type=memory_type,
            namespace=namespace,
            top_k=top_k,
            filters=filters
        )
    
    def get_context_window(
        self,
        namespace: str = "default",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Get conversation context for the agent.
        
        Args:
            namespace: User or session identifier
            max_tokens: Maximum tokens (uses config default if None)
        
        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or self.config.max_context_tokens
        return self.memory_store.get_context_window(
            namespace=namespace,
            max_tokens=max_tokens
        )
    
    def __repr__(self) -> str:
        return f"MongoAgent(name='{self.name}', model='{self.config.model}', persona='{self.config.persona}')"
