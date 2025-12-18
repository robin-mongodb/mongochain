"""Pydantic schemas for mongochain."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class Memory(BaseModel):
    """Base memory model for short-term and long-term memories."""
    
    id: Optional[str] = None
    content: str
    memory_type: Literal["semantic", "episodic", "procedural"]
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    namespace: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class Message(BaseModel):
    """Individual message in a conversation."""
    
    role: Literal["user", "assistant"]
    content: str
    embedding: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Conversation(BaseModel):
    """Conversation record with message history."""
    
    id: Optional[str] = None
    namespace: str
    agent_name: str
    conversation_id: str
    messages: List[Message] = Field(default_factory=list)
    summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserProfile(BaseModel):
    """User profile with consolidated long-term memories."""
    
    namespace: str
    name: Optional[str] = None
    role: Optional[str] = None
    company: Optional[str] = None
    timezone: Optional[str] = None
    memories: List[Memory] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Episode(BaseModel):
    """Past interaction episode for learning."""
    
    id: Optional[str] = None
    namespace: str
    agent_name: str
    interaction_summary: str
    success_score: float = 0.0
    approach_used: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentConfig(BaseModel):
    """Configuration for MongoAgent."""
    
    name: str
    connection_string: str
    model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"
    persona: Optional[str] = None
    collaborator_agents: List[str] = Field(default_factory=list)
    enable_short_term: bool = True
    enable_long_term: bool = True
    short_term_ttl_days: int = 90
    max_context_tokens: int = 4000
    openai_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None


class SearchResult(BaseModel):
    """Memory search result."""
    
    memory: Memory
    score: float
