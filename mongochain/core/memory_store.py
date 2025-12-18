"""MongoMemoryStore - MongoDB backend for agent memory."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure
from mongochain.core.schemas import SearchResult


class MongoMemoryStore:
    """MongoDB storage backend for agent memories."""
    
    CONVERSATIONS_COLLECTION = "conversations"
    SHORT_TERM_COLLECTION = "short_term_memory"
    USER_PROFILES_COLLECTION = "user_profiles"
    EPISODES_COLLECTION = "episodes"
    
    def __init__(
        self,
        connection_string: str,
        database_name: str,
        embedding_model: str = "text-embedding-3-small",
        short_term_ttl_days: int = 90
    ):
        """Initialize MongoMemoryStore."""
        self.connection_string = connection_string
        self.database_name = database_name
        self.embedding_model = embedding_model
        self.short_term_ttl_days = short_term_ttl_days
        
        self.client = None
        self.db = None
        self._connect()
        self._initialize_collections()
        
        print(f"  ✓ Database '{database_name}' initialized")
        print(f"  ✓ Collections: {self.CONVERSATIONS_COLLECTION}, {self.SHORT_TERM_COLLECTION}, {self.USER_PROFILES_COLLECTION}, {self.EPISODES_COLLECTION}")
    
    def _connect(self):
        """Establish MongoDB connection."""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            print(f"  ✓ Connected to MongoDB")
        except ConnectionFailure as e:
            raise ConnectionError(f"Failed to connect: {e}")
    
    def _initialize_collections(self):
        """Create collections and indexes."""
        # Conversations collection
        conversations = self.db[self.CONVERSATIONS_COLLECTION]
        conversations.create_index([("namespace", ASCENDING)])
        conversations.create_index([("agent_name", ASCENDING)])
        conversations.create_index([("created_at", DESCENDING)])
        conversations.create_index([("conversation_id", ASCENDING)], unique=True)
        
        # Short-term memory with TTL (90 days)
        short_term = self.db[self.SHORT_TERM_COLLECTION]
        short_term.create_index(
            [("created_at", ASCENDING)],
            expireAfterSeconds=self.short_term_ttl_days * 24 * 3600,
            name="ttl_index"
        )
        short_term.create_index([("namespace", ASCENDING), ("created_at", DESCENDING)])
        short_term.create_index([("memory_type", ASCENDING)])
        
        # User profiles (consolidated with long-term memories)
        user_profiles = self.db[self.USER_PROFILES_COLLECTION]
        user_profiles.create_index([("namespace", ASCENDING)], unique=True)
        user_profiles.create_index([("memories.importance", DESCENDING)])
        user_profiles.create_index([("updated_at", DESCENDING)])
        
        # Episodes
        episodes = self.db[self.EPISODES_COLLECTION]
        episodes.create_index([("namespace", ASCENDING)])
        episodes.create_index([("agent_name", ASCENDING)])
        episodes.create_index([("created_at", DESCENDING)])
        episodes.create_index([("success_score", DESCENDING)])
        
        print(f"  ✓ Indexes created")
    
    def store_memory(
        self,
        content: str,
        memory_type: str,
        namespace: str,
        metadata: Dict[str, Any],
        importance: float = 0.5
    ) -> str:
        """Store a new memory."""
        memory_doc = {
            "content": content,
            "memory_type": memory_type,
            "namespace": namespace,
            "metadata": metadata,
            "importance": importance,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "access_count": 0,
            "embedding": None
        }
        
        # High importance memories go to user_profiles as long-term memory
        if importance > 0.7 or memory_type == "semantic":
            # Add to user_profiles.memories array
            self.db[self.USER_PROFILES_COLLECTION].update_one(
                {"namespace": namespace},
                {
                    "$push": {"memories": memory_doc},
                    "$set": {"updated_at": datetime.utcnow()},
                    "$setOnInsert": {
                        "namespace": namespace,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return f"user_profile_{namespace}"
        else:
            # Short-term memory
            result = self.db[self.SHORT_TERM_COLLECTION].insert_one(memory_doc)
            return str(result.inserted_id)
    
    def search_memory(
        self,
        query: str,
        memory_type: Optional[str] = None,
        namespace: str = "default",
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search memories (placeholder - will implement vector search)."""
        return []
    
    def get_context_window(
        self,
        namespace: str,
        max_tokens: int
    ) -> str:
        """Get conversation context (placeholder)."""
        return ""
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
