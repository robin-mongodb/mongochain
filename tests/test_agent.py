"""Tests for MongoAgent."""

import pytest
from mongochain import MongoAgent, set_connection_string


class TestAgentNameValidation:
    """Test agent name validation."""
    
    def test_valid_agent_name(self):
        """Test that valid agent names pass."""
        agent = MongoAgent(name="support_agent")
        assert agent.name == "support_agent"
    
    def test_agent_name_with_spaces(self):
        """Test that spaces are converted to underscores."""
        agent = MongoAgent(name="my support agent")
        assert "_" in agent.name or " " not in agent.name
    
    def test_agent_name_with_invalid_chars(self):
        """Test that invalid characters are removed."""
        agent = MongoAgent(name="agent/with.invalid$chars")
        assert "/" not in agent.name
        assert "." not in agent.name
        assert "$" not in agent.name
    
    def test_agent_name_too_long(self):
        """Test that names longer than 63 chars are truncated."""
        long_name = "a" * 100
        agent = MongoAgent(name=long_name)
        assert len(agent.name) <= 63
    
    def test_agent_name_system_prefix(self):
        """Test that 'system.' prefix raises error."""
        with pytest.raises(ValueError):
            MongoAgent(name="system.test")
    
    def test_empty_agent_name(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError):
            MongoAgent(name="")


class TestAgentConfiguration:
    """Test agent configuration."""
    
    def test_default_configuration(self):
        """Test default config values."""
        agent = MongoAgent(name="test_agent")
        assert agent.config.model == "gpt-4"
        assert agent.config.embedding_model == "text-embedding-3-small"
        assert agent.config.short_term_ttl_days == 90
        assert agent.config.max_context_tokens == 4000
    
    def test_custom_configuration(self):
        """Test custom config values."""
        agent = MongoAgent(
            name="test_agent",
            model="claude-3-sonnet",
            embedding_model="voyage-3",
            persona="Technical expert",
            short_term_ttl_days=60,
            max_context_tokens=8000
        )
        assert agent.config.model == "claude-3-sonnet"
        assert agent.config.embedding_model == "voyage-3"
        assert agent.config.persona == "Technical expert"
        assert agent.config.short_term_ttl_days == 60
        assert agent.config.max_context_tokens == 8000
    
    def test_collaborator_agents(self):
        """Test collaborator agents configuration."""
        collaborators = ["agent_1", "agent_2", "agent_3"]
        agent = MongoAgent(
            name="test_agent",
            collaborator_agents=collaborators
        )
        assert agent.config.collaborator_agents == collaborators


class TestAgentMethods:
    """Test agent methods (requires MongoDB)."""
    
    def test_store_memory_placeholder(self):
        """Test that store_memory is callable."""
        agent = MongoAgent(name="test_agent")
        # This will use MongoDB if configured
        result = agent.store_memory(
            content="Test memory",
            memory_type="semantic",
            namespace="test_user",
            importance=0.8
        )
        # Should return a memory ID
        assert result is not None
    
    def test_search_memory_placeholder(self):
        """Test that search_memory returns a list."""
        agent = MongoAgent(name="test_agent")
        results = agent.search_memory(
            query="test query",
            namespace="test_user"
        )
        # Should return a list (empty for now with placeholder)
        assert isinstance(results, list)
    
    def test_get_context_window_placeholder(self):
        """Test that get_context_window returns string."""
        agent = MongoAgent(name="test_agent")
        context = agent.get_context_window(namespace="test_user")
        # Should return a string
        assert isinstance(context, str)


class TestAgentRepr:
    """Test agent string representation."""
    
    def test_agent_repr(self):
        """Test __repr__ method."""
        agent = MongoAgent(
            name="test_agent",
            model="gpt-4",
            persona="Test persona"
        )
        repr_str = repr(agent)
        assert "MongoAgent" in repr_str
        assert "test_agent" in repr_str
        assert "gpt-4" in repr_str


@pytest.fixture
def agent():
    """Fixture to create a test agent."""
    return MongoAgent(name="test_agent_fixture")


def test_agent_fixture(agent):
    """Test using agent fixture."""
    assert agent.name == "test_agent_fixture"
