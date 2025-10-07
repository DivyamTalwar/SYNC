import os
import time
from typing import Optional, List, Dict, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.utils.logging import get_logger
from config.base import config

logger = get_logger("openrouter")


class OpenRouterClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or config.api.openrouter_api_key
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided")

        self.default_model = default_model or config.api.primary_model
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api/v1"

        self.client = httpx.AsyncClient(timeout=timeout)
        self.sync_client = httpx.Client(timeout=timeout)

        logger.info(f"Initialized OpenRouter client with model: {default_model}")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://sync-system.ai",
            "X-Title": "SYNC Multi-Agent System",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completion asynchronously

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to default_model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Response dictionary
        """
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()

        result = response.json()
        logger.debug(f"Generated completion with {model}")

        return result

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completion synchronously

        Args:
            messages: List of message dicts
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Response dictionary
        """
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        try:
            response = self.sync_client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Generated completion with {model}")

            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from response

        Args:
            response: API response

        Returns:
            Generated text content
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract content from response: {e}")
            return ""

    async def generate_reasoning(
        self,
        query: str,
        context: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> str:
        """
        Generate agent reasoning for a query

        Args:
            query: Task or query to reason about
            context: Optional context information
            agent_role: Optional role description for the agent

        Returns:
            Reasoning text
        """
        system_prompt = "You are an expert reasoning agent in a multi-agent collaborative system."
        if agent_role:
            system_prompt += f" Your role: {agent_role}"

        user_message = f"Task: {query}"
        if context:
            user_message += f"\n\nContext: {context}"

        user_message += "\n\nProvide detailed reasoning about this task, including your approach, considerations, and any potential issues or alternative solutions."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = await self.generate_async(messages, temperature=0.7)
        return self.extract_content(response)

    async def generate_message(
        self,
        objective: str,
        own_reasoning: str,
        target_gap_info: Optional[str] = None,
    ) -> str:
        """
        Generate a strategic communication message

        Args:
            objective: Communication objective
            own_reasoning: Agent's own reasoning
            target_gap_info: Information about gaps with target

        Returns:
            Generated message
        """
        system_prompt = "You are generating a strategic communication message in a multi-agent collaboration."

        user_message = f"""Communication Objective: {objective}

Your Reasoning:
{own_reasoning}"""

        if target_gap_info:
            user_message += f"\n\nGap Analysis:\n{target_gap_info}"

        user_message += "\n\nGenerate a concise, strategic message that achieves the objective. Be direct and clear."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = await self.generate_async(messages, temperature=0.6, max_tokens=512)
        return self.extract_content(response)

    async def close(self):
        """Close client connections"""
        await self.client.aclose()
        self.sync_client.close()
        logger.info("Closed OpenRouter client")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sync_client.close()


# Global client instance
_client: Optional[OpenRouterClient] = None


def get_client() -> OpenRouterClient:
    """Get or create global OpenRouter client"""
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client
