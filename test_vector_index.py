"""Test script to verify vector index creation."""

import mongochain

# Set MongoDB connection string
CONNECTION_STRING = ""  # Add your connection string here
mongochain.set_connection_string(CONNECTION_STRING)

# Create an agent
print("Creating agent...")
agent = mongochain.MongoAgent(
    name="test_vector_index_agent",
    model="gpt-4",
    embedding_model="text-embedding-3-small"
)

print("\n" + "="*60)
print("Vector Index Creation Test Complete")
print("="*60)
print("\nTo verify indexes were created:")
print("1. Go to MongoDB Atlas UI")
print("2. Navigate to your cluster")
print("3. Go to 'Collections' > 'Search Indexes'")
print("4. Look for indexes named 'vector_index' on:")
print("   - short_term_memory collection")
print("   - user_profiles collection")
print("\nOr run this in MongoDB shell:")
print("  db.short_term_memory.listSearchIndexes()")
print("  db.user_profiles.listSearchIndexes()")

