from abc import ABC, abstractmethod
import logging
from typing import List
from langchain_core.messages import HumanMessage

from app.schemas.config import settings
from app.llm.prompts import PromptBuilder

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, context: List[str]) -> str:
        """
        Generate a response given a user prompt and retrieved context chunks.
        """
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai is required for OpenAIProvider")
            
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing")
            
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model="gpt-3.5-turbo", # Default fast model
            temperature=0
        )
        
    async def generate(self, prompt: str, context: List[str]) -> str:
        final_prompt = PromptBuilder.build_prompt(prompt, context)
        response = await self.llm.ainvoke([HumanMessage(content=final_prompt)])
        return response.content

class AnthropicProvider(BaseLLMProvider):
    def __init__(self):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("langchain-anthropic is required for AnthropicProvider")
            
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is missing")
            
        self.llm = ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model="claude-3-haiku-20240307",
            temperature=0
        )
        
    async def generate(self, prompt: str, context: List[str]) -> str:
        final_prompt = PromptBuilder.build_prompt(prompt, context)
        response = await self.llm.ainvoke([HumanMessage(content=final_prompt)])
        return response.content

class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("langchain-google-genai is required for GeminiProvider")
            
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing")
            
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=settings.gemini_api_key,
            model="gemini-1.5-flash",
            temperature=0
        )
        
    async def generate(self, prompt: str, context: List[str]) -> str:
        final_prompt = PromptBuilder.build_prompt(prompt, context)
        response = await self.llm.ainvoke([HumanMessage(content=final_prompt)])
        return response.content

class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        try:
            # pyrefly: ignore [missing-import]
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("langchain-ollama is required for OllamaProvider")
            
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0
        )
        
    async def generate(self, prompt: str, context: List[str]) -> str:
        final_prompt = PromptBuilder.build_prompt(prompt, context)
        response = await self.llm.ainvoke([HumanMessage(content=final_prompt)])
        return response.content

class GroqProvider(BaseLLMProvider):
    def __init__(self):
        try:
            # pyrefly: ignore [missing-import]
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("langchain-groq is required for GroqProvider")
            
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing")
            
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0
        )
        
    async def generate(self, prompt: str, context: List[str]) -> str:
        final_prompt = PromptBuilder.build_prompt(prompt, context)
        response = await self.llm.ainvoke([HumanMessage(content=final_prompt)])
        return response.content

def get_llm_provider() -> BaseLLMProvider:
    """
    Factory function to instantiate the active LLM provider based on settings.
    """
    provider_name = settings.llm_provider.lower()
    
    try:
        if provider_name == "openai":
            return OpenAIProvider()
        elif provider_name == "anthropic":
            return AnthropicProvider()
        elif provider_name == "gemini":
            return GeminiProvider()
        elif provider_name == "ollama":
            return OllamaProvider()
        elif provider_name == "groq":
            return GroqProvider()
        else:
            logger.warning(f"Unknown LLM_PROVIDER '{provider_name}'. Falling back to OpenAI.")
            return OpenAIProvider()
    except Exception as e:
        logger.error(f"Failed to initialize LLM provider '{provider_name}': {e}")
        raise RuntimeError(f"LLM Configuration Error: {e}")
