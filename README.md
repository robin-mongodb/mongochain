# Mongochain

A simple Python agent framework that uses **MongoDB Atlas** as the memory layer, demonstrating how MongoDB can power stateful AI agents with user-specific memories.

## Features

- **User-Specific Memory**: All memories tied to user email/ID
- **Three Memory Types**:
  - `conversation_history` - Raw chat log with embeddings (90-day TTL)
  - `short_term_memory` - Session summaries (7-day TTL)
  - `long_term_memory` - Persistent user facts (no TTL, vector search)
- **Auto-Extraction**: LLM automatically extracts and stores user facts
- **Dynamic Personas**: Change agent personality on the fly
- **Multi-Provider LLM**: Works with OpenAI, Anthropic, and Google
- **Multi-Agent Collaboration**: Agents can share user memories

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

# Create an agent
agent = MongoAgent(
    name="assistant",
    persona="You are a helpful research assistant.",
    mongo_uri="mongodb+srv://...",
    voyage_api_key="voy-...",
    llm_api_key="sk-..."
)

# Chat requires a user_id (email)
response = agent.chat(
    user_id="user@example.com",
    message="Hi! I'm a DevOps engineer working on MongoDB optimization."
)
print(response)

# The agent automatically extracts and stores user facts!
# Check what it learned:
memories = agent.get_user_memories("user@example.com")
for m in memories:
    print(f"- {m['content']}")
```

## Memory Architecture

Each agent creates a MongoDB database with three collections:

| Collection             | Purpose                  | TTL     | Features              |
| ---------------------- | ------------------------ | ------- | --------------------- |
| `conversation_history` | Raw chat with embeddings | 90 days | Vector search enabled |
| `short_term_memory`    | Session summaries        | 7 days  | Vector search enabled |
| `long_term_memory`     | Persistent user facts    | None    | Vector search enabled |

All memories are **user-specific** (filtered by `user_id`).

## Storing User Memories

### Automatic (via LLM)

The agent automatically extracts significant facts from conversations:

```python
# User mentions their job - automatically stored
agent.chat("user@example.com", "I'm a senior engineer at Acme Corp")
```

### Manual

```python
# Store a specific fact
agent.store_user_memory(
    user_id="user@example.com",
    content="User prefers detailed technical explanations",
    category="preference"
)
```

## Dynamic Personas

```python
# Change persona on the fly
agent.set_persona("You are a grumpy professor who gives brief answers.")
response = agent.chat("user@example.com", "Explain indexing")

agent.set_persona("You are a pirate who explains things with nautical metaphors.")
response = agent.chat("user@example.com", "Explain indexing")  # Different style!
```

## Multi-Agent Collaboration

```python
# Create Alice - a research specialist
alice = MongoAgent(
    name="alice",
    persona="You are a research specialist.",
    mongo_uri=MONGO_URI,
    voyage_api_key=VOYAGE_KEY,
    llm_api_key=LLM_KEY
)

# Create Bob - can access Alice's memories
bob = MongoAgent(
    name="bob",
    persona="You are a summarization expert.",
    mongo_uri=MONGO_URI,
    voyage_api_key=VOYAGE_KEY,
    llm_api_key=LLM_KEY,
    collaborators=["alice"]  # Bob can read Alice's user memories
)

# Bob can now access what Alice knows about users
response = bob.chat("user@example.com", "What has Alice learned about me?")
```

## Using Different LLM Providers

```python
# OpenAI (default)
agent = MongoAgent(name="assistant", llm_provider="openai", ...)

# Anthropic Claude
agent = MongoAgent(
    name="assistant",
    llm_provider="anthropic",
    llm_api_key="sk-ant-...",
    ...
)

# Google Gemini
agent = MongoAgent(
    name="assistant",
    llm_provider="google",
    llm_api_key="AIza...",
    llm_model="gemini-1.5-pro",  # Optional: override default model
    ...
)
```

## Session Management

```python
# End session to save summary to short-term memory
agent.end_session("user@example.com")

# Get user stats
stats = agent.get_user_stats("user@example.com")
print(f"Conversations: {stats['conversation_count']}")
print(f"Short-term summaries: {stats['short_term_count']}")
print(f"Long-term facts: {stats['long_term_count']}")

# Clear user memories
agent.clear_user_memories("user@example.com", memory_type="conversation")
```

## Requirements

- Python 3.10+
- MongoDB Atlas cluster (for vector search support)
- API keys for:
  - Voyage AI (embeddings)
  - Your chosen LLM provider (OpenAI, Anthropic, or Google)

## License

MIT
