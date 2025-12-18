"""Global configuration for MongoChain."""

from typing import Optional


class MongoChainConfig:
    """Global configuration manager for MongoChain."""
    
    _connection_string: Optional[str] = None
    
    @classmethod
    def set_connection_string(cls, connection_string: str) -> None:
        """
        Set the MongoDB connection string globally.
        
        Args:
            connection_string: MongoDB connection URI
            
        Raises:
            ValueError: If connection_string is empty
        """
        if not connection_string:
            raise ValueError("Connection string cannot be empty")
        cls._connection_string = connection_string
        print(f"✓ MongoChain connection string configured")
    
    @classmethod
    def get_connection_string(cls) -> str:
        """
        Get the MongoDB connection string.
        
        Returns:
            Connection string
            
        Raises:
            RuntimeError: If connection string has not been set
        """
        if cls._connection_string is None:
            raise RuntimeError(
                "MongoDB connection string not configured. "
                "Call mongochain.set_connection_string() first."
            )
        return cls._connection_string


# Convenience function for users
def set_connection_string(connection_string: str) -> None:
    """Set MongoDB connection string globally."""
    MongoChainConfig.set_connection_string(connection_string)
