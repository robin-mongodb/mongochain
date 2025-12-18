"""Tests for MongoMemoryStore."""

import pytest
from mongochain import MongoMemoryStore, set_connection_string
from datetime import datetime


class TestMemoryStoreInitialization:
    """Test MongoMemoryStore initialization."""
    
    def test_memory_store_creates_collections(self):
        """Test that memory store creates all required collections."""
        store = MongoMemoryStore(
            connection_string="mongodb://localhost:27017",
            database_name="test_db"
        )
        # Check that all collections exist
        collections = store.db.list_collection_names()
        assert "conversations" in collections
        assert "short_term_memory" in collections
        assert "user_profiles" in collections
        assert "episodes" in collections
    
    def test_memory_store_creates_indexes(self):
        """Test that memory store creates indexes."""
        store = MongoMemoryStore(
            connection_string="mongodb://localhost:27017",
            database_name="test_db_indexes"
        )
        # Check conversations indexes
        conversations_indexes = store.db["conversations"].index_information()
        assert len(conversations_indexes) > 1  # At least default _id index + custom indexes
    
    def test_ttl_configuration(self):
        """Test TTL configuration for short-term memory."""
        store = MongoMemoryStore(
            connection_string="mongodb://localhost:27017",
            database_name="test_db_ttl",
            short_term_ttl_days=30
        )
        # Check TTL index
        indexes = store.db["short_term_memory"].index_information()
        assert "ttl_index" in indexes


class TestMemoryStorage:
    """Test memory storage operations."""
    
    def test_store_short_term_memory(self):
        """Test storing short-term memory."""
        store = MongoMemoryStore(
            connection_string="mongodb://localhost:27017",
            database_name="test_store_short"
        )
        memory_id = store.store_memory(
            content="This is a short-term memory",
            memory_type="episodic",
            namespace="test_user",
            metadata={"source": "test"},
            importance=0.3
        )
        assert memory_id is not None
        
        # Verify it was stored
        doc = store.db["short_term_memory"].find_one({"_id": memory_id})
        assert doc is not None
        assert doc["content"] == "This is a short-term memory"
    
    def test_store_long_term_memory(self):
        """Test storing long-term memory in user profile."""
        store = MongoMemoryStore(
            connection_string="mongodb://localhost:27017",
            database_name="test_store_long"
        )
        memory_id = store.store_memory(
            content="This is a long-term memory",
            memory_type="semantic",
            namespace="test_user",
            metadata={"importance_level": "high"},
            importance=0.9
        )
        assert memory_id is not None
        
        # Verify it was stored in user_profiles
        user = store.db["user_profiles"].find_one({"namespace": "test_user"})
        assert user is not None
        assert len(user["memories"]) > 0
    
    def test_store_memory_with_metadata(self):
        """Test storing memory with additional metadata."""
        store = MongoMemoryStore(
            connection_string="mongodb://localhost:27017",
            database_name="test_store_metadata"
        )
        metadata = {
            "source": "conversation",
            "context": "onboarding",
            "tags": ["important", "user-preference"]
        }
        memory_id = store.store_memory(
            content="User prefers Slack notifications",
            memory_type="semantic",
            namespace="test_user",
            metadata=metadata,
            importance=0.7
        )
        assert memory_id is not None


class TestMemorySearch:
    """Test memory search operations."""
    
    def test_search_memory_returns_list(self):
        """Test that search_memory returns a list."""
        store = MongoMemoryStore(
            connection_string="mongodb://localhost:27017",
            database_name="test_search"
        )
        results = store.search_memory(
            query="test query",
            namespace="test_user"
        )
        # Should return list (empty with placeholder implementation)
        assert isinstance(results, list)
    
    def test_search_memory_with_filters(self):
        """Test search_memory with filters."""
        store = MongoMemoryStore(
            connection_string="mongodb://localhost:27017",
            database_name="test_search_filters"
        )
        results = store.search_memory(
            query="test",
            namespace="test_user",
            memory_type="semantic",
            top_k=5,
            filters={"importance": {"$gte": 0.5}}
        )
        assert isinstance(results, list)


class TestContextWindow:
    """Test context window retrieval."""
    
    def test_get_context_window_returns_string(self):
        """Test that get_context_window returns string."""
        store = MongoMemoryStore(
            connection_string="mongodb://localhost:27017",
            database_name="test_context"
        )
        context = store.get_context_window(
            namespace="test_user",
            max_tokens=4000
        )
        # Should return string (empty with placeholder implementation)
        assert isinstance(context, str)


class TestCollectionNames:
    """Test collection name constants."""
    
    def test_collection_names(self):
        """Test that collection names are correctly defined."""
        assert MongoMemoryStore.CONVERSATIONS_COLLECTION == "conversations"
        assert MongoMemoryStore.SHORT_TERM_COLLECTION == "short_term_memory"
        assert MongoMemoryStore.USER_PROFILES_COLLECTION == "user_profiles"
        assert MongoMemoryStore.EPISODES_COLLECTION == "episodes"


@pytest.fixture
def memory_store():
    """Fixture to create a test memory store."""
    return MongoMemoryStore(
        connection_string="mongodb://localhost:27017",
        database_name="test_fixture_db"
    )


def test_memory_store_fixture(memory_store):
    """Test using memory store fixture."""
    assert memory_store.database_name == "test_fixture_db"
    assert memory_store.db is not None
