"""Embedding utilities for MongoChain."""

from typing import List, Optional
import os


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        raise NotImplementedError


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model: OpenAI embedding model (text-embedding-3-small, text-embedding-3-large)
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding


class VoyageEmbeddings(EmbeddingProvider):
    """Voyage AI embedding provider."""
    
    def __init__(self, model: str = "voyage-3", api_key: Optional[str] = None):
        """
        Initialize Voyage AI embeddings.
        
        Args:
            model: Voyage embedding model (voyage-3, voyage-lite-02)
            api_key: Voyage API key (uses VOYAGE_API_KEY env var if not provided)
        """
        self.model = model
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Voyage API key not found. "
                "Set VOYAGE_API_KEY environment variable or pass api_key parameter."
            )
        
        try:
            import voyageai
            self.client = voyageai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("voyageai package required. Install with: pip install voyageai")
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text using Voyage AI.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        result = self.client.embed([text], model=self.model)
        return result.embeddings[0]


def get_embedding_provider(model: str, api_key: Optional[str] = None) -> EmbeddingProvider:
    """
    Get appropriate embedding provider based on model name.
    
    Args:
        model: Model name (e.g., "text-embedding-3-small", "voyage-3")
        api_key: API key for the provider
    
    Returns:
        EmbeddingProvider instance
    
    Raises:
        ValueError: If model is not recognized
    """
    if "voyage" in model.lower():
        return VoyageEmbeddings(model=model, api_key=api_key)
    elif "text-embedding" in model.lower():
        return OpenAIEmbeddings(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown embedding model: {model}")
