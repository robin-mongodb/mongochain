"""Memory storage using MongoDB Atlas for mongochain."""

from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import OperationFailure

from .config import MemoryConfig


class MemoryStore:
    """Handles all memory operations with MongoDB.
    
    Manages three types of memory:
    - short_term_memory: Recent context with TTL auto-expiry
    - long_term_memory: Persistent facts with vector embeddings
    - conversation_history: Full chat log
    
    Attributes:
        db_name: Name of the MongoDB database
        config: Memory configuration settings
    """
    
    # Collection names
    SHORT_TERM = "short_term_memory"
    LONG_TERM = "long_term_memory"
    CONVERSATION = "conversation_history"
    
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        config: Optional[MemoryConfig] = None
    ):
        """Initialize the memory store.
        
        Args:
            mongo_uri: MongoDB Atlas connection string
            db_name: Name for the database (typically agent name)
            config: Optional memory configuration
        """
        self.db_name = db_name
        self.config = config or MemoryConfig()
        
        # Connect to MongoDB
        self._client = MongoClient(mongo_uri)
        self._db = self._client[db_name]
        
        # Get collection references
        self._short_term = self._db[self.SHORT_TERM]
        self._long_term = self._db[self.LONG_TERM]
        self._conversation = self._db[self.CONVERSATION]
    
    def setup_collections(self) -> dict:
        """Create collections and indexes.
        
        Returns:
            Dict with status of each collection/index creation
        """
        status = {
            "collections_created": [],
            "indexes_created": [],
            "vector_index_status": None
        }
        
        # Ensure collections exist by inserting and deleting a doc
        for coll_name in [self.SHORT_TERM, self.LONG_TERM, self.CONVERSATION]:
            if coll_name not in self._db.list_collection_names():
                self._db.create_collection(coll_name)
                status["collections_created"].append(coll_name)
        
        # Create TTL index on short_term_memory
        self._short_term.create_index(
            "created_at",
            expireAfterSeconds=self.config.short_term_ttl_seconds
        )
        status["indexes_created"].append(f"{self.SHORT_TERM}.created_at (TTL)")
        
        # Create timestamp index on conversation_history
        self._conversation.create_index("timestamp")
        status["indexes_created"].append(f"{self.CONVERSATION}.timestamp")
        
        # Create timestamp index on long_term_memory
        self._long_term.create_index("timestamp")
        status["indexes_created"].append(f"{self.LONG_TERM}.timestamp")
        
        # Create vector search index on long_term_memory
        status["vector_index_status"] = self._create_vector_index()
        
        return status
    
    def _create_vector_index(self) -> str:
        """Create Atlas Vector Search index on long_term_memory.
        
        Returns:
            Status message for the vector index creation
        """
        index_name = "vector_index"
        
        # Check if index already exists
        try:
            existing_indexes = list(self._long_term.list_search_indexes())
            for idx in existing_indexes:
                if idx.get("name") == index_name:
                    return f"Vector index '{index_name}' already exists"
        except OperationFailure:
            # list_search_indexes may not be available on all deployments
            pass
        
        # Create the vector search index
        try:
            self._long_term.create_search_index({
                "definition": {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": self.config.embedding_dimensions,
                                "similarity": "cosine"
                            }
                        }
                    }
                },
                "name": index_name
            })
            return f"Vector index '{index_name}' created (may take a moment to build)"
        except OperationFailure as e:
            if "already exists" in str(e).lower():
                return f"Vector index '{index_name}' already exists"
            return f"Vector index creation note: {str(e)}"
    
    # ==================== Short-Term Memory ====================
    
    def store_short_term(self, content: str, metadata: Optional[dict] = None) -> str:
        """Store content in short-term memory.
        
        Args:
            content: The content to store
            metadata: Optional additional metadata
            
        Returns:
            The inserted document ID as string
        """
        doc = {
            "content": content,
            "created_at": datetime.now(timezone.utc),
            "metadata": metadata or {}
        }
        result = self._short_term.insert_one(doc)
        return str(result.inserted_id)
    
    def get_recent_context(self, limit: Optional[int] = None) -> list[dict]:
        """Retrieve recent short-term memories.
        
        Args:
            limit: Max number of memories to retrieve
            
        Returns:
            List of memory documents
        """
        limit = limit or self.config.short_term_limit
        cursor = self._short_term.find().sort("created_at", -1).limit(limit)
        return list(cursor)
    
    # ==================== Long-Term Memory ====================
    
    def store_long_term(
        self,
        content: str,
        embedding: list[float],
        metadata: Optional[dict] = None
    ) -> str:
        """Store content with embedding in long-term memory.
        
        Args:
            content: The content to store
            embedding: Vector embedding of the content
            metadata: Optional additional metadata
            
        Returns:
            The inserted document ID as string
        """
        doc = {
            "content": content,
            "embedding": embedding,
            "timestamp": datetime.now(timezone.utc),
            "metadata": metadata or {}
        }
        result = self._long_term.insert_one(doc)
        return str(result.inserted_id)
    
    def search_long_term(
        self,
        query_embedding: list[float],
        limit: Optional[int] = None
    ) -> list[dict]:
        """Search long-term memory using vector similarity.
        
        Args:
            query_embedding: Vector embedding of the query
            limit: Max number of results to return
            
        Returns:
            List of matching documents with similarity scores
        """
        limit = limit or self.config.long_term_limit
        
        # Use Atlas Vector Search aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "content": 1,
                    "timestamp": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            results = list(self._long_term.aggregate(pipeline))
            return results
        except OperationFailure as e:
            # Vector index may not be ready yet
            if "index not found" in str(e).lower():
                # Fallback to recent memories
                cursor = self._long_term.find().sort("timestamp", -1).limit(limit)
                return list(cursor)
            raise
    
    def get_all_long_term(self, limit: int = 100) -> list[dict]:
        """Get all long-term memories (for collaboration access).
        
        Args:
            limit: Max number of memories to retrieve
            
        Returns:
            List of memory documents
        """
        cursor = self._long_term.find(
            {},
            {"embedding": 0}  # Exclude embeddings for efficiency
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    
    # ==================== Conversation History ====================
    
    def store_conversation(self, role: str, content: str) -> str:
        """Store a message in conversation history.
        
        Args:
            role: Message role ("user" or "assistant")
            content: Message content
            
        Returns:
            The inserted document ID as string
        """
        doc = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc)
        }
        result = self._conversation.insert_one(doc)
        return str(result.inserted_id)
    
    def get_conversation_history(self, limit: Optional[int] = None) -> list[dict]:
        """Retrieve recent conversation history.
        
        Args:
            limit: Max number of messages to retrieve
            
        Returns:
            List of message documents in chronological order
        """
        limit = limit or self.config.conversation_limit
        cursor = self._conversation.find().sort("timestamp", -1).limit(limit)
        # Reverse to get chronological order
        messages = list(cursor)
        messages.reverse()
        return messages
    
    # ==================== Utility Methods ====================
    
    def clear_short_term(self):
        """Clear all short-term memories."""
        self._short_term.delete_many({})
    
    def clear_conversation(self):
        """Clear conversation history."""
        self._conversation.delete_many({})
    
    def clear_all(self):
        """Clear all memories (use with caution)."""
        self._short_term.delete_many({})
        self._long_term.delete_many({})
        self._conversation.delete_many({})
    
    def get_stats(self) -> dict:
        """Get statistics about stored memories.
        
        Returns:
            Dict with counts for each memory type
        """
        return {
            "short_term_count": self._short_term.count_documents({}),
            "long_term_count": self._long_term.count_documents({}),
            "conversation_count": self._conversation.count_documents({})
        }
    
    def close(self):
        """Close the MongoDB connection."""
        self._client.close()
