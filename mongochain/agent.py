"""Main MongoAgent class for mongochain."""

from typing import Optional

from .config import AgentConfig, MemoryConfig
from .embeddings import VoyageEmbeddings
from .llm import LLMClient
from .memory import MemoryStore


class MongoAgent:
    """An AI agent with MongoDB-backed memory.
    
    Each agent gets its own MongoDB database with separate collections for
    different memory types (short-term, long-term, conversation history).
    
    Attributes:
        name: Agent name
        persona: Current agent personality/system prompt
        collaborators: List of agent names this agent can access
    """
    
    def __init__(
        self,
        name: str,
        persona: str,
        mongo_uri: str,
        voyage_api_key: str,
        llm_api_key: str,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        collaborators: Optional[list[str]] = None,
        memory_config: Optional[MemoryConfig] = None
    ):
        """Initialize a MongoAgent.
        
        Args:
            name: Agent name (becomes the MongoDB database name)
            persona: The agent's personality/system prompt
            mongo_uri: MongoDB Atlas connection string
            voyage_api_key: Voyage AI API key for embeddings
            llm_api_key: API key for the LLM provider
            llm_provider: LLM provider ("openai", "anthropic", or "google")
            llm_model: Specific model to use (uses provider default if None)
            collaborators: List of agent names this agent can read memories from
            memory_config: Optional memory configuration
        """
        # Store configuration
        self._config = AgentConfig(
            name=name,
            persona=persona,
            mongo_uri=mongo_uri,
            voyage_api_key=voyage_api_key,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider,
            llm_model=llm_model,
            collaborators=collaborators or []
        )
        
        self.name = name
        self.persona = persona
        self.collaborators = list(self._config.collaborators)
        self._mongo_uri = mongo_uri
        
        # Initialize components
        self._embeddings = VoyageEmbeddings(api_key=voyage_api_key)
        self._llm = LLMClient(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model
        )
        self._memory = MemoryStore(
            mongo_uri=mongo_uri,
            db_name=self._config.db_name,
            config=memory_config
        )
        
        # Setup database and collections
        self._setup()
    
    def _setup(self):
        """Setup the agent's database and collections."""
        status = self._memory.setup_collections()
        
        # Print confirmation message
        print(f"Agent: {self.name} created. Check database '{self._config.db_name}' in your MongoDB cluster.")
        
        if status["collections_created"]:
            print(f"  Collections created: {', '.join(status['collections_created'])}")
        if status["vector_index_status"]:
            print(f"  {status['vector_index_status']}")
    
    def chat(self, message: str) -> str:
        """Send a message and get a response.
        
        The agent will:
        1. Store the message in conversation history
        2. Retrieve relevant context from all memory types
        3. Include collaborator memories if authorized
        4. Generate a response using the LLM
        5. Store the response and update memories
        
        Args:
            message: The user's message
            
        Returns:
            The agent's response
        """
        # Store user message in conversation history
        self._memory.store_conversation("user", message)
        
        # Store in short-term memory
        self._memory.store_short_term(f"User said: {message}")
        
        # Build context from memories
        context = self._build_context(message)
        
        # Build messages for LLM
        messages = self._build_messages(message, context)
        
        # Get response from LLM
        response = self._llm.chat(messages, system_prompt=self._build_system_prompt())
        
        # Store assistant response
        self._memory.store_conversation("assistant", response)
        self._memory.store_short_term(f"Assistant said: {response}")
        
        # Extract and store any important information in long-term memory
        self._update_long_term_memory(message, response)
        
        return response
    
    def _build_context(self, query: str) -> dict:
        """Build context from all memory sources.
        
        Args:
            query: The current query to find relevant memories
            
        Returns:
            Dict with context from each memory type
        """
        context = {
            "short_term": [],
            "long_term": [],
            "conversation": [],
            "collaborator_memories": {}
        }
        
        # Get short-term context
        short_term = self._memory.get_recent_context()
        context["short_term"] = [m["content"] for m in short_term]
        
        # Get relevant long-term memories via vector search
        try:
            query_embedding = self._embeddings.embed_query(query)
            long_term = self._memory.search_long_term(query_embedding)
            context["long_term"] = [m["content"] for m in long_term]
        except Exception:
            # Fallback if vector search fails
            pass
        
        # Get conversation history
        conversation = self._memory.get_conversation_history()
        context["conversation"] = [
            {"role": m["role"], "content": m["content"]}
            for m in conversation
        ]
        
        # Get collaborator memories
        for collab_name in self.collaborators:
            try:
                collab_memories = self._get_collaborator_memories(collab_name, query)
                if collab_memories:
                    context["collaborator_memories"][collab_name] = collab_memories
            except Exception:
                # Skip if collaborator DB doesn't exist
                pass
        
        return context
    
    def _get_collaborator_memories(self, agent_name: str, query: str) -> list[str]:
        """Get relevant memories from a collaborator agent.
        
        Args:
            agent_name: Name of the collaborator agent
            query: Query to find relevant memories
            
        Returns:
            List of relevant memory contents
        """
        # Create a temporary memory store for the collaborator
        collab_config = AgentConfig(
            name=agent_name,
            persona="",
            mongo_uri=self._mongo_uri,
            voyage_api_key="",
            llm_api_key="",
        )
        
        collab_memory = MemoryStore(
            mongo_uri=self._mongo_uri,
            db_name=collab_config.db_name
        )
        
        try:
            # Try vector search first
            query_embedding = self._embeddings.embed_query(query)
            memories = collab_memory.search_long_term(query_embedding, limit=3)
            return [m["content"] for m in memories]
        except Exception:
            # Fallback to recent memories
            memories = collab_memory.get_all_long_term(limit=3)
            return [m["content"] for m in memories]
        finally:
            collab_memory.close()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return f"""You are an AI agent named {self.name}.

{self.persona}

You have access to memories from past conversations and can recall relevant information.
When referencing information from your memories, integrate it naturally into your responses."""
    
    def _build_messages(self, current_message: str, context: dict) -> list[dict]:
        """Build the message list for the LLM.
        
        Args:
            current_message: The current user message
            context: Context from memory retrieval
            
        Returns:
            List of message dicts for the LLM
        """
        messages = []
        
        # Add context as a system-like message if we have relevant memories
        context_parts = []
        
        if context["long_term"]:
            context_parts.append(
                "Relevant memories:\n" + 
                "\n".join(f"- {m}" for m in context["long_term"])
            )
        
        if context["collaborator_memories"]:
            for collab_name, memories in context["collaborator_memories"].items():
                if memories:
                    context_parts.append(
                        f"Memories from {collab_name}:\n" +
                        "\n".join(f"- {m}" for m in memories)
                    )
        
        if context_parts:
            messages.append({
                "role": "user",
                "content": "[Context from memory]\n" + "\n\n".join(context_parts)
            })
            messages.append({
                "role": "assistant",
                "content": "I understand. I'll use this context to inform my response."
            })
        
        # Add recent conversation history (skip the current message we just stored)
        for msg in context["conversation"][:-1]:  # Exclude the message we just added
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": current_message
        })
        
        return messages
    
    def _update_long_term_memory(self, user_message: str, assistant_response: str):
        """Extract and store important information in long-term memory.
        
        This is a simple implementation that stores significant exchanges.
        A more sophisticated version could use the LLM to extract key facts.
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
        """
        # Store exchanges that seem important (longer responses often contain info)
        if len(assistant_response) > 200:
            # Create a summary of the exchange
            summary = f"User asked about: {user_message[:100]}... Response covered: {assistant_response[:200]}..."
            embedding = self._embeddings.embed(summary)
            self._memory.store_long_term(
                content=summary,
                embedding=embedding,
                metadata={
                    "type": "exchange_summary",
                    "user_message_preview": user_message[:100]
                }
            )
    
    def set_persona(self, new_persona: str):
        """Change the agent's persona dynamically.
        
        Args:
            new_persona: The new personality/system prompt
        """
        self.persona = new_persona
        self._config.persona = new_persona
        print(f"Agent {self.name}'s persona updated.")
    
    def add_collaborator(self, agent_name: str):
        """Add an agent as a collaborator.
        
        Args:
            agent_name: Name of the agent to add as collaborator
        """
        if agent_name not in self.collaborators:
            self.collaborators.append(agent_name)
            print(f"Added {agent_name} as a collaborator for {self.name}.")
    
    def remove_collaborator(self, agent_name: str):
        """Remove an agent from collaborators.
        
        Args:
            agent_name: Name of the agent to remove
        """
        if agent_name in self.collaborators:
            self.collaborators.remove(agent_name)
            print(f"Removed {agent_name} from collaborators for {self.name}.")
    
    def store_memory(self, content: str, memory_type: str = "long_term"):
        """Manually store a memory.
        
        Args:
            content: Content to store
            memory_type: Type of memory ("short_term" or "long_term")
        """
        if memory_type == "short_term":
            self._memory.store_short_term(content)
        else:
            embedding = self._embeddings.embed(content)
            self._memory.store_long_term(content, embedding)
        print(f"Memory stored in {memory_type}.")
    
    def get_memory_stats(self) -> dict:
        """Get statistics about the agent's memories.
        
        Returns:
            Dict with memory counts
        """
        return self._memory.get_stats()
    
    def clear_memories(self, memory_type: Optional[str] = None):
        """Clear memories.
        
        Args:
            memory_type: Type to clear ("short_term", "conversation", or None for all)
        """
        if memory_type == "short_term":
            self._memory.clear_short_term()
        elif memory_type == "conversation":
            self._memory.clear_conversation()
        elif memory_type is None:
            self._memory.clear_all()
        print(f"Cleared {memory_type or 'all'} memories.")
    
    def __repr__(self) -> str:
        return f"MongoAgent(name='{self.name}', provider='{self._llm.provider}')"
