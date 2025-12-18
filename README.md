# MongoChain

MongoDB-backed agent memory framework inspired by LangChain.

## Installation

```bash
pip install git+https://github.com/yourusername/mongochain.git
```

## Quick Start

```python
from mongochain import MongoAgent

# Create an agent with MongoDB memory
agent = MongoAgent(
    name="customer_support_bot",
    connection_string="mongodb+srv://your-connection-string",
    model="gpt-4"
)

# Agent automatically manages its own database and collections
```

## Features

- **One Database Per Agent**: Complete memory isolation
- **Automatic Scaffolding**: Collections created automatically
- **Vector Search**: Semantic memory search with MongoDB Atlas
- **Short & Long-term Memory**: Conversation context + persistent knowledge
