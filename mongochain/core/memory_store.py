"""MongoMemoryStore - MongoDB backend for agent memory."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
from mongochain.core.schemas import SearchResult
from mongochain.utils.embeddings import get_embedding_dimensions


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
        
        # Create vector search indexes
        self._create_vector_indexes()
    
    def _create_vector_indexes(self):
        """
        Create vector search indexes for collections that store embeddings.
        
        Creates vector indexes for:
        - short_term_memory: on 'embedding' field
        - user_profiles: on 'memories.embedding' field (nested array)
        """
        try:
            # Get embedding dimensions based on model
            dimensions = get_embedding_dimensions(self.embedding_model)
            print(f"  → Creating vector indexes (model: {self.embedding_model}, dimensions: {dimensions})")
            
            # Create vector index for short_term_memory collection
            self._create_vector_index_for_collection(
                collection_name=self.SHORT_TERM_COLLECTION,
                index_name="vector_index",
                field_path="embedding",
                dimensions=dimensions
            )
            
            # Create vector index for user_profiles collection (nested in memories array)
            self._create_vector_index_for_collection(
                collection_name=self.USER_PROFILES_COLLECTION,
                index_name="vector_index",
                field_path="memories.embedding",
                dimensions=dimensions
            )
            
            print(f"  ✓ Vector indexes created successfully")
            
            # Verify indexes were created
            self._verify_vector_indexes(dimensions)
        except Exception as e:
            # Don't fail initialization if vector index creation fails
            # (e.g., if not using Atlas or vector search not available)
            print(f"  ⚠ Vector index creation failed: {type(e).__name__}: {e}")
            print(f"    (Vector search requires MongoDB Atlas M10+ with vector search enabled)")
            print(f"    (MongoDB version 7.0+ required)")
            print(f"    You can verify indexes in Atlas UI: Collections > Search Indexes")
    
    def _verify_vector_indexes(self, expected_dimensions: int):
        """Verify that vector indexes were created successfully."""
        try:
            for collection_name in [self.SHORT_TERM_COLLECTION, self.USER_PROFILES_COLLECTION]:
                collection = self.db[collection_name]
                try:
                    indexes = list(collection.list_search_indexes())
                    vector_indexes = [idx for idx in indexes if idx.get("name") == "vector_index"]
                    if vector_indexes:
                        idx = vector_indexes[0]
                        status = idx.get("status", "unknown")
                        print(f"    → Verified '{collection_name}': status={status}")
                    else:
                        print(f"    ⚠ No vector index found on '{collection_name}'")
                except Exception as e:
                    print(f"    ⚠ Could not verify indexes on '{collection_name}': {e}")
        except Exception:
            # Silently fail verification - not critical
            pass
    
    def _create_vector_index_for_collection(
        self,
        collection_name: str,
        index_name: str,
        field_path: str,
        dimensions: int
    ):
        """
        Create a vector search index for a collection.
        
        Args:
            collection_name: Name of the collection
            index_name: Name for the vector index
            field_path: Path to the embedding field (e.g., "embedding" or "memories.embedding")
            dimensions: Number of dimensions in the embedding vectors
        """
        try:
            collection = self.db[collection_name]
            
            # Check if index already exists
            try:
                existing_indexes = list(collection.list_search_indexes())
                for idx in existing_indexes:
                    if idx.get("name") == index_name:
                        # Index already exists, skip creation
                        print(f"    ✓ Vector index '{index_name}' already exists on {collection_name}")
                        return
            except Exception as list_error:
                # If list_search_indexes fails, try to create anyway
                print(f"    ⚠ Could not list existing indexes: {list_error}")
            
            # Create vector search index definition
            # Using the correct format for MongoDB Atlas Vector Search
            # Format: https://www.mongodb.com/docs/atlas/atlas-search/create-index/
            index_definition = {
                "name": index_name,
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            field_path: {
                                "type": "knnVector",
                                "dimensions": dimensions,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
            }
            
            # Create the vector search index
            # Note: This requires MongoDB Atlas with vector search enabled
            # MongoDB Atlas Vector Search uses the Search API, not regular indexes
            
            # Try using collection.create_search_index() method (pymongo 4.5+)
            # This is the recommended way for Atlas Search indexes
            if hasattr(collection, 'create_search_index'):
                try:
                    result = collection.create_search_index(index_definition)
                    print(f"    ✓ Creating vector index '{index_name}' on {collection_name}")
                    print(f"      Field: {field_path}, Dimensions: {dimensions}")
                    if result:
                        print(f"      Index ID: {result}")
                    return
                except Exception as method_error:
                    error_msg = str(method_error)
                    # If it's a known error, provide helpful message
                    if "not authorized" in error_msg.lower():
                        raise Exception(f"Not authorized to create search indexes. Check Atlas permissions.")
                    elif "atlas" in error_msg.lower() or "search" in error_msg.lower():
                        raise Exception(f"Atlas Search API error: {error_msg}")
                    raise
            
            # Fallback: Use database command (for older pymongo or alternative setups)
            # Format: { "createSearchIndexes": "collection_name", "indexes": [...] }
            try:
                result = self.db.command({
                    "createSearchIndexes": collection_name,
                    "indexes": [index_definition]
                })
                print(f"    ✓ Creating vector index '{index_name}' on {collection_name} (using command)")
                print(f"      Field: {field_path}, Dimensions: {dimensions}")
                if result:
                    print(f"      Command result: {result}")
                return
            except OperationFailure as cmd_error:
                error_msg = str(cmd_error)
                error_lower = error_msg.lower()
                
                if "unknown command" in error_lower:
                    raise Exception(
                        f"createSearchIndexes command not recognized. "
                        f"This might mean:\n"
                        f"  1. You're not using MongoDB Atlas\n"
                        f"  2. Your MongoDB version doesn't support vector search (needs 7.0+)\n"
                        f"  3. Vector search is not enabled on your cluster"
                    )
                elif "not authorized" in error_lower or "unauthorized" in error_lower:
                    raise Exception(f"Not authorized to create search indexes. Check database user permissions.")
                elif "atlas" in error_lower:
                    raise Exception(f"Atlas-specific error: {error_msg}")
                else:
                    raise Exception(f"Failed to create vector index: {error_msg}")
            except Exception as e:
                raise Exception(f"Unexpected error creating vector index: {type(e).__name__}: {e}")
        except OperationFailure as e:
            # Handle case where vector search is not available
            error_msg = str(e)
            if "vector" in error_msg.lower() or "search" in error_msg.lower() or "atlas" in error_msg.lower():
                raise Exception(f"Vector search not available: {error_msg}")
            # Re-raise other operation failures
            raise
        except Exception as e:
            # Catch any other errors and provide helpful message
            error_msg = str(e)
            if "create_search_index" in error_msg or "createSearchIndex" in error_msg:
                raise Exception(f"Failed to create vector index: {error_msg}")
            raise
    
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
