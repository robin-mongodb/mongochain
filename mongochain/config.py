"""Configuration classes for mongochain."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for a MongoAgent.
    
    Attributes:
        name: Agent name (becomes the MongoDB database name)
        persona: The agent's personality/system prompt
        mongo_uri: MongoDB Atlas connection string
        voyage_api_key: Voyage AI API key for embeddings
        llm_api_key: API key for the chosen LLM provider
        llm_provider: LLM provider ("openai", "anthropic", or "google")
        llm_model: Specific model to use (uses provider default if None)
        collaborators: List of agent names this agent can read memories from
    """
    name: str
    persona: str
    mongo_uri: str
    voyage_api_key: str
    llm_api_key: str
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    collaborators: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_providers = {"openai", "anthropic", "google"}
        if self.llm_provider not in valid_providers:
            raise ValueError(
                f"Invalid llm_provider '{self.llm_provider}'. "
                f"Must be one of: {', '.join(valid_providers)}"
            )
        
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        
        if not self.mongo_uri:
            raise ValueError("MongoDB URI cannot be empty")
        
        # Sanitize agent name for use as database name
        self.db_name = self._sanitize_db_name(self.name)
    
    @staticmethod
    def _sanitize_db_name(name: str) -> str:
        """Sanitize agent name for use as MongoDB database name.
        
        MongoDB database names cannot contain: /\\. "$*<>:|?
        and should be lowercase for consistency.
        """
        invalid_chars = '/\\. "$*<>:|?'
        sanitized = name.lower()
        for char in invalid_chars:
            sanitized = sanitized.replace(char, "_")
        return sanitized


@dataclass
class MemoryConfig:
    """Configuration for memory storage.
    
    Attributes:
        short_term_ttl_seconds: TTL for short-term memories (default: 1 hour)
        short_term_limit: Max short-term memories to retrieve
        long_term_limit: Max long-term memories to retrieve via vector search
        conversation_limit: Max conversation messages to include in context
        embedding_dimensions: Dimensions of the embedding vectors
    """
    short_term_ttl_seconds: int = 3600  # 1 hour
    short_term_limit: int = 10
    long_term_limit: int = 5
    conversation_limit: int = 20
    embedding_dimensions: int = 1024  # voyage-3-lite default
