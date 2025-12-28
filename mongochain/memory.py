"""Memory storage using MongoDB Atlas for mongochain."""

from datetime import datetime, timezone, timedelta
from typing import Optional

from pymongo import MongoClient, ASCENDING
from pymongo.errors import OperationFailure

from .config import MemoryConfig


class MemoryStore:
    """Handles all memory operations with MongoDB.
    
    Manages three types of memory:
    - conversation_history: Raw chat log with embeddings (90-day TTL)
    - short_term_memory: Session summaries per user (7-day TTL)
    - long_term_memory: Persistent user-specific facts with vector search
    
    All memories are user-specific, identified by user_id (email).
    
    Attributes:
        db_name: Name of the MongoDB database
        config: Memory configuration settings
    """
    
    # Collection names
    CONVERSATION = "conversation_history"
    SHORT_TERM = "short_term_memory"
    LONG_TERM = "long_term_memory"
    AGENT_METADATA = "agent_metadata"
    
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
        self._conversation = self._db[self.CONVERSATION]
        self._short_term = self._db[self.SHORT_TERM]
        self._long_term = self._db[self.LONG_TERM]
        self._agent_metadata = self._db[self.AGENT_METADATA]
    
    def setup_collections(self) -> dict:
        """Create collections and indexes.
        
        Returns:
            Dict with status of each collection/index creation
        """
        status = {
            "collections_created": [],
            "indexes_created": [],
            "vector_index_status": []
        }
        
        # Ensure collections exist
        for coll_name in [self.CONVERSATION, self.SHORT_TERM, self.LONG_TERM, self.AGENT_METADATA]:
            if coll_name not in self._db.list_collection_names():
                self._db.create_collection(coll_name)
                status["collections_created"].append(coll_name)
        
        # === Conversation History Indexes ===
        # TTL index (90 days)
        self._conversation.create_index(
            "expires_at",
            expireAfterSeconds=0  # Expire at the specified date
        )
        status["indexes_created"].append(f"{self.CONVERSATION}.expires_at (TTL)")
        
        # Compound index for user + timestamp queries
        self._conversation.create_index([
            ("user_id", ASCENDING),
            ("timestamp", ASCENDING)
        ])
        status["indexes_created"].append(f"{self.CONVERSATION}.user_id+timestamp")
        
        # Vector index for conversation semantic search
        vector_status = self._create_vector_index(
            self._conversation, 
            "conversation_vector_index"
        )
        status["vector_index_status"].append(f"conversation_history: {vector_status}")
        
        # === Short-Term Memory Indexes ===
        # TTL index (7 days)
        self._short_term.create_index(
            "expires_at",
            expireAfterSeconds=0
        )
        status["indexes_created"].append(f"{self.SHORT_TERM}.expires_at (TTL)")
        
        # User + timestamp index
        self._short_term.create_index([
            ("user_id", ASCENDING),
            ("created_at", ASCENDING)
        ])
        status["indexes_created"].append(f"{self.SHORT_TERM}.user_id+created_at")
        
        # Vector index for short-term semantic search
        vector_status = self._create_vector_index(
            self._short_term,
            "short_term_vector_index"
        )
        status["vector_index_status"].append(f"short_term_memory: {vector_status}")
        
        # === Long-Term Memory Indexes ===
        # User + timestamp index
        self._long_term.create_index([
            ("user_id", ASCENDING),
            ("timestamp", ASCENDING)
        ])
        status["indexes_created"].append(f"{self.LONG_TERM}.user_id+timestamp")
        
        # Vector index for long-term semantic search
        vector_status = self._create_vector_index(
            self._long_term,
            "long_term_vector_index"
        )
        status["vector_index_status"].append(f"long_term_memory: {vector_status}")
        
        return status
    
    def _create_vector_index(self, collection, index_name: str) -> str:
        """Create Atlas Vector Search index on a collection.
        
        Args:
            collection: The MongoDB collection
            index_name: Name for the vector index
            
        Returns:
            Status message for the vector index creation
        """
        # Check if index already exists
        try:
            existing_indexes = list(collection.list_search_indexes())
            for idx in existing_indexes:
                if idx.get("name") == index_name:
                    return f"'{index_name}' already exists"
        except OperationFailure:
            pass
        
        # Create the vector search index
        try:
            collection.create_search_index({
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
            return f"'{index_name}' created"
        except OperationFailure as e:
            if "already exists" in str(e).lower():
                return f"'{index_name}' already exists"
            return f"'{index_name}' note: {str(e)}"
    
    # ==================== Conversation History ====================
    
    def store_conversation(
        self,
        user_id: str,
        role: str,
        content: str,
        embedding: list[float]
    ) -> str:
        """Store a message in conversation history with embedding.
        
        Args:
            user_id: User's email/ID
            role: Message role ("user" or "assistant")
            content: Message content
            embedding: Vector embedding of the content
            
        Returns:
            The inserted document ID as string
        """
        expires_at = datetime.now(timezone.utc) + timedelta(days=self.config.conversation_ttl_days)
        
        doc = {
            "user_id": user_id,
            "role": role,
            "content": content,
            "embedding": embedding,
            "timestamp": datetime.now(timezone.utc),
            "expires_at": expires_at
        }
        result = self._conversation.insert_one(doc)
        return str(result.inserted_id)
    
    def get_conversation_history(
        self,
        user_id: str,
        limit: Optional[int] = None
    ) -> list[dict]:
        """Retrieve recent conversation history for a user.
        
        Args:
            user_id: User's email/ID
            limit: Max number of messages to retrieve
            
        Returns:
            List of message documents in chronological order
        """
        limit = limit or self.config.conversation_limit
        cursor = self._conversation.find(
            {"user_id": user_id},
            {"embedding": 0}  # Exclude embedding for efficiency
        ).sort("timestamp", -1).limit(limit)
        
        messages = list(cursor)
        messages.reverse()  # Chronological order
        return messages
    
    def search_conversations(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 5
    ) -> list[dict]:
        """Search conversation history using vector similarity.
        
        Args:
            user_id: User's email/ID
            query_embedding: Vector embedding of the query
            limit: Max results to return
            
        Returns:
            List of matching messages with similarity scores
        """
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "conversation_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 20,
                    "limit": limit * 2,
                    "filter": {"user_id": user_id}
                }
            },
            {
                "$project": {
                    "user_id": 1,
                    "role": 1,
                    "content": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
            {"$limit": limit}
        ]
        
        try:
            return list(self._conversation.aggregate(pipeline))
        except OperationFailure:
            return []
    
    # ==================== Short-Term Memory ====================
    
    def store_short_term(
        self,
        user_id: str,
        summary: str,
        topics_discussed: list[str],
        actions_taken: list[str],
        embedding: list[float],
        session_id: Optional[str] = None
    ) -> str:
        """Store a session summary in short-term memory.
        
        Args:
            user_id: User's email/ID
            summary: Summary of what was discussed
            topics_discussed: List of topics covered
            actions_taken: List of actions/outcomes
            embedding: Vector embedding of the summary
            session_id: Optional session identifier
            
        Returns:
            The inserted document ID as string
        """
        expires_at = datetime.now(timezone.utc) + timedelta(days=self.config.short_term_ttl_days)
        
        doc = {
            "user_id": user_id,
            "summary": summary,
            "topics_discussed": topics_discussed,
            "actions_taken": actions_taken,
            "embedding": embedding,
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc),
            "expires_at": expires_at
        }
        result = self._short_term.insert_one(doc)
        return str(result.inserted_id)
    
    def get_recent_short_term(
        self,
        user_id: str,
        limit: Optional[int] = None
    ) -> list[dict]:
        """Get recent short-term summaries for a user.
        
        Args:
            user_id: User's email/ID
            limit: Max summaries to retrieve
            
        Returns:
            List of summary documents
        """
        limit = limit or self.config.short_term_limit
        cursor = self._short_term.find(
            {"user_id": user_id},
            {"embedding": 0}
        ).sort("created_at", -1).limit(limit)
        return list(cursor)
    
    def search_short_term(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: Optional[int] = None
    ) -> list[dict]:
        """Search short-term memory using vector similarity.
        
        Args:
            user_id: User's email/ID
            query_embedding: Vector embedding of the query
            limit: Max results to return
            
        Returns:
            List of matching summaries with scores
        """
        limit = limit or self.config.short_term_limit
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "short_term_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": {"user_id": user_id}
                }
            },
            {
                "$project": {
                    "user_id": 1,
                    "summary": 1,
                    "topics_discussed": 1,
                    "actions_taken": 1,
                    "created_at": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            return list(self._short_term.aggregate(pipeline))
        except OperationFailure:
            # Fallback to recent memories
            return self.get_recent_short_term(user_id, limit)
    
    # ==================== Long-Term Memory ====================
    
    def store_long_term(
        self,
        user_id: str,
        content: str,
        embedding: list[float],
        category: str = "general",
        metadata: Optional[dict] = None
    ) -> str:
        """Store a user-specific fact in long-term memory.
        
        Args:
            user_id: User's email/ID
            content: The fact/information to store
            embedding: Vector embedding of the content
            category: Category of the memory (e.g., "preference", "fact", "skill")
            metadata: Optional additional metadata
            
        Returns:
            The inserted document ID as string
        """
        doc = {
            "user_id": user_id,
            "content": content,
            "embedding": embedding,
            "category": category,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc)
        }
        result = self._long_term.insert_one(doc)
        return str(result.inserted_id)
    
    def search_long_term(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: Optional[int] = None,
        category: Optional[str] = None
    ) -> list[dict]:
        """Search long-term memory using vector similarity.
        
        Args:
            user_id: User's email/ID
            query_embedding: Vector embedding of the query
            limit: Max results to return
            category: Optional category filter
            
        Returns:
            List of matching memories with scores
        """
        limit = limit or self.config.long_term_limit
        
        # Build filter
        filter_doc = {"user_id": user_id}
        if category:
            filter_doc["category"] = category
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "long_term_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": filter_doc
                }
            },
            {
                "$project": {
                    "user_id": 1,
                    "content": 1,
                    "category": 1,
                    "metadata": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            return list(self._long_term.aggregate(pipeline))
        except OperationFailure:
            # Fallback to recent memories
            return self.get_user_long_term(user_id, limit)
    
    def get_user_long_term(
        self,
        user_id: str,
        limit: int = 10,
        category: Optional[str] = None
    ) -> list[dict]:
        """Get all long-term memories for a user.
        
        Args:
            user_id: User's email/ID
            limit: Max memories to retrieve
            category: Optional category filter
            
        Returns:
            List of memory documents
        """
        query = {"user_id": user_id}
        if category:
            query["category"] = category
            
        cursor = self._long_term.find(
            query,
            {"embedding": 0}
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    
    # ==================== Agent Metadata ====================
    
    def save_agent_metadata(
        self,
        name: str,
        persona: str,
        llm_provider: str,
        llm_model: str,
        collaborators: list[str],
        description: Optional[str] = None
    ) -> dict:
        """Save or update agent metadata.
        
        Uses upsert to create on first call, update on subsequent calls.
        
        Args:
            name: Agent name
            persona: Agent personality/system prompt
            llm_provider: LLM provider name
            llm_model: LLM model name
            collaborators: List of collaborator agent names
            description: Optional human-readable description
            
        Returns:
            The saved metadata document
        """
        now = datetime.now(timezone.utc)
        
        # Check if metadata already exists
        existing = self._agent_metadata.find_one({"_id": "agent_config"})
        
        doc = {
            "_id": "agent_config",
            "name": name,
            "persona": persona,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "collaborators": collaborators,
            "description": description,
            "updated_at": now
        }
        
        if existing:
            # Preserve created_at on update
            doc["created_at"] = existing.get("created_at", now)
        else:
            doc["created_at"] = now
        
        self._agent_metadata.replace_one(
            {"_id": "agent_config"},
            doc,
            upsert=True
        )
        
        return doc
    
    def get_agent_metadata(self) -> Optional[dict]:
        """Retrieve agent metadata.
        
        Returns:
            The agent metadata document, or None if not found
        """
        return self._agent_metadata.find_one({"_id": "agent_config"})
    
    def update_agent_persona(self, persona: str) -> bool:
        """Update only the agent's persona.
        
        Args:
            persona: New persona/system prompt
            
        Returns:
            True if updated, False if no metadata exists
        """
        result = self._agent_metadata.update_one(
            {"_id": "agent_config"},
            {
                "$set": {
                    "persona": persona,
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        return result.modified_count > 0
    
    def update_agent_collaborators(self, collaborators: list[str]) -> bool:
        """Update the agent's collaborators list.
        
        Args:
            collaborators: New list of collaborator agent names
            
        Returns:
            True if updated, False if no metadata exists
        """
        result = self._agent_metadata.update_one(
            {"_id": "agent_config"},
            {
                "$set": {
                    "collaborators": collaborators,
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        return result.modified_count > 0
    
    # ==================== Utility Methods ====================
    
    def clear_user_conversation(self, user_id: str):
        """Clear conversation history for a user."""
        self._conversation.delete_many({"user_id": user_id})
    
    def clear_user_short_term(self, user_id: str):
        """Clear short-term memories for a user."""
        self._short_term.delete_many({"user_id": user_id})
    
    def clear_user_long_term(self, user_id: str):
        """Clear long-term memories for a user."""
        self._long_term.delete_many({"user_id": user_id})
    
    def clear_user_all(self, user_id: str):
        """Clear all memories for a user."""
        self.clear_user_conversation(user_id)
        self.clear_user_short_term(user_id)
        self.clear_user_long_term(user_id)
    
    def clear_all(self):
        """Clear all memories for all users (use with caution)."""
        self._conversation.delete_many({})
        self._short_term.delete_many({})
        self._long_term.delete_many({})
    
    def get_user_stats(self, user_id: str) -> dict:
        """Get memory statistics for a user.
        
        Args:
            user_id: User's email/ID
            
        Returns:
            Dict with counts for each memory type
        """
        return {
            "conversation_count": self._conversation.count_documents({"user_id": user_id}),
            "short_term_count": self._short_term.count_documents({"user_id": user_id}),
            "long_term_count": self._long_term.count_documents({"user_id": user_id})
        }
    
    def get_stats(self) -> dict:
        """Get overall memory statistics.
        
        Returns:
            Dict with total counts and unique users
        """
        return {
            "total_conversations": self._conversation.count_documents({}),
            "total_short_term": self._short_term.count_documents({}),
            "total_long_term": self._long_term.count_documents({}),
            "unique_users": len(self._conversation.distinct("user_id"))
        }
    
    def close(self):
        """Close the MongoDB connection."""
        self._client.close()
