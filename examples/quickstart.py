"""MongoChain Quick Start Example."""

import mongochain

# Set MongoDB connection string (do this once at the start)
CONNECTION_STRING = ""
mongochain.set_connection_string(CONNECTION_STRING)

# Create an agent with MongoDB memory
agent = mongochain.MongoAgent(
    name="customer_support_bot",
    model="gpt-4",
    embedding_model="text-embedding-3-small",
    persona="Helpful customer support specialist",
    openai_api_key="your-openai-key"  # or use OPENAI_API_KEY env var
)

print(f"\n{agent}\n")

# Store some memories
print("Storing memories...")
memory_id_1 = agent.store_memory(
    content="User prefers email communication over phone calls",
    memory_type="semantic",
    namespace="user_12345",
    importance=0.9
)
print(f"✓ Stored memory: {memory_id_1}")

memory_id_2 = agent.store_memory(
    content="User works in DevOps and uses MongoDB Atlas",
    memory_type="semantic",
    namespace="user_12345",
    importance=0.8
)
print(f"✓ Stored memory: {memory_id_2}")

memory_id_3 = agent.store_memory(
    content="User recently asked about vector search capabilities",
    memory_type="episodic",
    namespace="user_12345",
    importance=0.6
)
print(f"✓ Stored memory: {memory_id_3}")

# Search memories (will implement vector search later)
print("\nSearching memories...")
results = agent.search_memory(
    query="What does the user prefer for communication?",
    namespace="user_12345",
    top_k=5
)
print(f"Found {len(results)} memories")

# Get context window (will implement later)
print("\nGetting context window...")
context = agent.get_context_window(namespace="user_12345")
print(f"Context: {context if context else '(will be populated with vector search)'}")

print("\n✓ Quick start complete!")
