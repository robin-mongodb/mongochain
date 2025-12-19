"""Voyage AI embeddings wrapper for mongochain."""

import voyageai


class VoyageEmbeddings:
    """Wrapper for Voyage AI embedding API.
    
    Provides a simple interface for generating embeddings using Voyage AI models.
    
    Attributes:
        model: The Voyage AI model to use for embeddings
    """
    
    # Model dimensions for reference
    MODEL_DIMENSIONS = {
        "voyage-3-lite": 1024,
        "voyage-3": 1024,
        "voyage-large-2": 1536,
        "voyage-code-2": 1536,
        "voyage-2": 1024,
    }
    
    def __init__(self, api_key: str, model: str = "voyage-3-lite"):
        """Initialize the Voyage AI embeddings client.
        
        Args:
            api_key: Voyage AI API key
            model: Model to use for embeddings (default: voyage-3-lite)
        """
        self.model = model
        self.client = voyageai.Client(api_key=api_key)
        self.dimensions = self.MODEL_DIMENSIONS.get(model, 1024)
    
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        result = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="document"
        )
        return result.embeddings[0]
    
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query (optimized for search).
        
        Args:
            text: The query text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        result = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="query"
        )
        return result.embeddings[0]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="document"
        )
        return result.embeddings
