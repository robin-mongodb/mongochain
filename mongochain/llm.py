"""Multi-provider LLM abstraction for mongochain."""

from typing import Optional, Generator


class LLMClient:
    """Unified interface for multiple LLM providers.
    
    Supports OpenAI, Anthropic Claude, and Google Gemini with a consistent API.
    
    Attributes:
        provider: The LLM provider being used
        model: The specific model being used
    """
    
    PROVIDERS = {
        "openai": {"default_model": "gpt-4o-mini"},
        "anthropic": {"default_model": "claude-3-haiku-20240307"},
        "google": {"default_model": "gemini-1.5-flash"},
    }
    
    def __init__(self, provider: str, api_key: str, model: Optional[str] = None):
        """Initialize the LLM client.
        
        Args:
            provider: LLM provider ("openai", "anthropic", or "google")
            api_key: API key for the provider
            model: Specific model to use (uses provider default if None)
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Must be one of: {', '.join(self.PROVIDERS.keys())}"
            )
        
        self.provider = provider
        self.model = model or self.PROVIDERS[provider]["default_model"]
        self._api_key = api_key
        self._client = None
        
        # Initialize the appropriate client
        self._init_client()
    
    def _init_client(self):
        """Initialize the provider-specific client."""
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
            
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
            
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self.model)
    
    def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Send messages and return response text.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt to prepend
            
        Returns:
            The assistant's response text
        """
        if self.provider == "openai":
            return self._chat_openai(messages, system_prompt)
        elif self.provider == "anthropic":
            return self._chat_anthropic(messages, system_prompt)
        elif self.provider == "google":
            return self._chat_google(messages, system_prompt)
    
    def chat_stream(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Send messages and stream response text.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt to prepend
            
        Yields:
            Chunks of the assistant's response text
        """
        if self.provider == "openai":
            yield from self._stream_openai(messages, system_prompt)
        elif self.provider == "anthropic":
            yield from self._stream_anthropic(messages, system_prompt)
        elif self.provider == "google":
            yield from self._stream_google(messages, system_prompt)
    
    # ==================== OpenAI ====================
    
    def _chat_openai(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Handle chat for OpenAI."""
        all_messages = []
        
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        
        all_messages.extend(messages)
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=all_messages
        )
        
        return response.choices[0].message.content
    
    def _stream_openai(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream chat for OpenAI."""
        all_messages = []
        
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        
        all_messages.extend(messages)
        
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    # ==================== Anthropic ====================
    
    def _chat_anthropic(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Handle chat for Anthropic Claude."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=messages
        )
        
        return response.content[0].text
    
    def _stream_anthropic(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream chat for Anthropic Claude."""
        with self._client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    # ==================== Google ====================
    
    def _chat_google(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Handle chat for Google Gemini."""
        history = []
        
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        
        chat = self._client.start_chat(history=history)
        
        last_message = messages[-1]["content"] if messages else ""
        if system_prompt and not history:
            last_message = f"{system_prompt}\n\n{last_message}"
        
        response = chat.send_message(last_message)
        
        return response.text
    
    def _stream_google(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream chat for Google Gemini."""
        history = []
        
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        
        chat = self._client.start_chat(history=history)
        
        last_message = messages[-1]["content"] if messages else ""
        if system_prompt and not history:
            last_message = f"{system_prompt}\n\n{last_message}"
        
        response = chat.send_message(last_message, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
