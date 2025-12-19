# Mongochain

A simple Python agent framework that uses **MongoDB Atlas** as the memory layer, demonstrating how MongoDB can power stateful AI agents with multiple memory types.

## Features

- **Multiple Memory Types**: Short-term, long-term, and conversation history
- **Persona Management**: Create agents with personalities that can be changed dynamically
- **Multi-Provider LLM Support**: Works with OpenAI, Anthropic Claude, and Google Gemini
- **Vector Search**: Semantic memory retrieval using MongoDB Atlas Vector Search
- **Multi-Agent Collaboration**: Agents can share memories with authorized collaborators
- **Voyage AI Embeddings**: High-quality embeddings for semantic search

## Installation

```bash
pip install mongochain
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```python
from mongochain import MongoAgent

# Create an agent with OpenAI (default)
agent = MongoAgent(
    name="assistant",
    persona="You are a helpful research assistant.",
    mongo_uri="mongodb+srv://...",
    voyage_api_key="voy-...",
    llm_api_key="sk-..."
)
# Output: "Agent: assistant created. Check database 'assistant' in your MongoDB cluster."

# Chat with the agent
response = agent.chat("Hello! What can you help me with?")
print(response)

# Change persona dynamically
agent.set_persona("You are a grumpy professor who gives very brief answers.")
response = agent.chat("Explain machine learning")  # Different tone!
```

## Using Different LLM Providers

```python
# Anthropic Claude
agent = MongoAgent(
    name="claude_agent",
    persona="You are a coding assistant.",
    mongo_uri="mongodb+srv://...",
    voyage_api_key="voy-...",
    llm_api_key="sk-ant-...",
    llm_provider="anthropic"
)

# Google Gemini
agent = MongoAgent(
    name="gemini_agent",
    persona="You are a creative writer.",
    mongo_uri="mongodb+srv://...",
    voyage_api_key="voy-...",
    llm_api_key="AIza...",
    llm_provider="google",
    llm_model="gemini-1.5-pro"  # Optional: override default model
)
```

## Multi-Agent Collaboration

```python
# Create first agent
alice = MongoAgent(
    name="alice",
    persona="You are a research specialist.",
    mongo_uri="mongodb+srv://...",
    voyage_api_key="voy-...",
    llm_api_key="sk-..."
)

# Create second agent with access to Alice's memories
bob = MongoAgent(
    name="bob",
    persona="You are a summarization expert.",
    mongo_uri="mongodb+srv://...",
    voyage_api_key="voy-...",
    llm_api_key="sk-...",
    collaborators=["alice"]  # Bob can read Alice's long-term memories
)

# Bob can now reference Alice's knowledge
response = bob.chat("What has Alice been researching?")
```

## Memory Architecture

Each agent gets its own MongoDB database with three collections:

| Collection             | Purpose          | Features              |
| ---------------------- | ---------------- | --------------------- |
| `short_term_memory`    | Recent context   | TTL auto-expiry       |
| `long_term_memory`     | Persistent facts | Vector search enabled |
| `conversation_history` | Full chat log    | Timestamp indexed     |

## Requirements

- Python 3.10+
- MongoDB Atlas cluster (for vector search support)
- API keys for:
  - Voyage AI (embeddings)
  - Your chosen LLM provider (OpenAI, Anthropic, or Google)

## License

MIT
