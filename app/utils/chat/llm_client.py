"""
OpenRouter LLM Client for chat functionality with memory support.
"""
import os
from typing import List, Dict, Optional
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenRouter client."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        # Available free models
        self.available_models = {
            "GPT OSS 20B": "openai/gpt-oss-20b:free",
            "DeepSeek R1": "deepseek/deepseek-r1:free",
            "DeepSeek Chat V3": "deepseek/deepseek-chat-v3-0324:free",
            "Qwen 3 30B": "qwen/qwen3-30b-a3b:free",
            "Dolphin Mistral 24B": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
        }
        
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models."""
        return self.available_models
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "openai/gpt-oss-20b:free",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate a chat completion using OpenRouter API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier for OpenRouter
            temperature: Sampling temperature (0-1.5)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response text
        """
        try:
            extra_headers = {
                "HTTP-Referer": "https://streamlit-chat-app.local",
                "X-Title": "Streamlit Chat App",
            }
            
            completion_args = {
                "extra_headers": extra_headers,
                "extra_body": {},
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                completion_args["max_tokens"] = max_tokens
            
            if stream:
                completion_args["stream"] = True
                return self._handle_stream_response(completion_args)
            else:
                completion = self.client.chat.completions.create(**completion_args)
                return completion.choices[0].message.content
                
        except Exception as e:
            error_msg = f"API Error: {str(e)}"
            st.error(f"Error generating response: {error_msg}")
            return f"I encountered an error while processing your request: {error_msg}"
    
    def _handle_stream_response(self, completion_args: dict) -> str:
        """Handle streaming response from OpenRouter."""
        response_text = ""
        try:
            stream = self.client.chat.completions.create(**completion_args)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response_text += chunk.choices[0].delta.content
            return response_text
        except Exception as e:
            error_msg = f"Streaming API Error: {str(e)}"
            st.error(f"Error in streaming response: {error_msg}")
            return response_text or f"I encountered an error while streaming the response: {error_msg}"


class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 20):
        """Initialize conversation manager."""
        self.max_history = max_history
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        st.session_state.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Keep only the last max_history messages
        if len(st.session_state.conversation_history) > self.max_history:
            st.session_state.conversation_history = st.session_state.conversation_history[-self.max_history:]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return st.session_state.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        st.session_state.conversation_history = []
    
    def get_messages_for_api(self, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get messages formatted for API call.
        
        Args:
            system_prompt: Optional system prompt to prepend
            
        Returns:
            List of messages formatted for OpenRouter API
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(self.get_conversation_history())
        
        return messages


def initialize_llm_client() -> Optional[OpenRouterClient]:
    """Initialize and return LLM client if API key is available."""
    try:
        # Check for API key in various places
        api_key = None
        
        # Try environment variable first
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Try Streamlit secrets
        if not api_key and hasattr(st, 'secrets'):
            try:
                api_key = st.secrets.get("OPENROUTER_API_KEY")
            except:
                pass
        
        if not api_key:
            st.warning("OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable or add it to Streamlit secrets.")
            st.info("You can get a free API key at: https://openrouter.ai/")
            return None
        
        return OpenRouterClient(api_key)
        
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {str(e)}")
        return None
