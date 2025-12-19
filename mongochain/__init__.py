"""Mongochain - MongoDB Atlas powered agent framework.

A simple Python library that demonstrates MongoDB Atlas as a memory layer
for AI agents with multiple memory types and multi-agent collaboration.
"""

from .agent import MongoAgent
from .config import AgentConfig, MemoryConfig
from .embeddings import VoyageEmbeddings
from .llm import LLMClient
from .memory import MemoryStore

__version__ = "0.1.0"

__all__ = [
    "MongoAgent",
    "AgentConfig",
    "MemoryConfig",
    "VoyageEmbeddings",
    "LLMClient",
    "MemoryStore",
]
