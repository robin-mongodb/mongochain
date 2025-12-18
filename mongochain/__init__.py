"""MongoChain - MongoDB-backed agent memory framework."""

from mongochain.core.agent import MongoAgent
from mongochain.core.memory_store import MongoMemoryStore
from mongochain.core.schemas import (
    Memory,
    Message,
    Conversation,
    UserProfile,
    Episode,
    AgentConfig,
    SearchResult,
)
from mongochain.core.config import set_connection_string

__version__ = "0.1.0"

__all__ = [
    "MongoAgent",
    "MongoMemoryStore",
    "Memory",
    "Message",
    "Conversation",
    "UserProfile",
    "Episode",
    "AgentConfig",
    "SearchResult",
    "set_connection_string",
]
