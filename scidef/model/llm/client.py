"""
LLM services: async client for definition extraction.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from scidef.model.dataclass import (
    CacheModel,
    EmptyResponseError,
    LLMClientError,
    ProviderType,
)
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


class LLMClient(CacheModel):
    """Async-only OpenAI-compatible LLM client for definition extraction."""

    def __init__(
        self,
        provider_type: ProviderType,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        cache_dir: Optional[Path] = None,
        log_level: int = logging.INFO,
    ):
        self.provider_type = provider_type
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.requests_total = 0
        self.requests_cached = 0

        self.is_openrouter = "openrouter.ai" in base_url.lower()
        timeout_config = httpx.Timeout(
            connect=10,
            read=timeout,
            write=timeout,
            pool=timeout,
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_config,
        )

        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.setLevel(log_level)
        logger.debug(
            f"Initialized {provider_type.value} LLM client: {base_url}",
        )

    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, float]:
        """Generate text using the language model asynchronously."""
        start_time = time.time()
        self.requests_total += 1

        if self.requests_total % 64 == 0:
            logger.debug(
                f"Total LLM requests so far: {self.requests_total}, "
                f"cached: {self.requests_cached} "
                f"({(self.requests_cached / self.requests_total) * 100:.2f}%)",
            )

        current_model = model or self.model_name
        cached_response = self.load_from_cache(
            prompt,
            current_model,
            system_prompt,
            temperature,
            top_p,
            **kwargs,
        )
        if cached_response is not None:
            duration = time.time() - start_time
            self.requests_cached += 1
            return cached_response, duration

        try:
            api_params = self._prepare_api_params(
                prompt,
                current_model,
                temperature,
                top_p,
                system_prompt,
                **kwargs,
            )

            generated_text = await self._generate_with_retry(api_params)
            duration = time.time() - start_time

            logger.debug(
                f"LLM generation successful: {len(generated_text)} chars in {duration:.2f}s",
            )
            self.save_to_cache(
                prompt,
                generated_text,
                current_model,
                system_prompt,
                temperature,
                top_p,
                **kwargs,
            )
            return generated_text, duration

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"LLM generation failed after {duration:.2f}s: {e}"
            logger.error(error_msg)
            if isinstance(e, LLMClientError) and "filter" in str(e).lower():
                # Content filtered errors are not retried
                logger.warning(
                    f"Content was filtered by the LLM API. The prompt was: {prompt}",
                )
            raise LLMClientError(error_msg, details={"duration": duration})

    async def _generate_with_retry(self, api_params: Dict[str, Any]) -> str:
        """Generate text with tenacity retry logic."""

        def _should_retry(exception: BaseException) -> bool:
            """Determine if an exception should trigger a retry."""
            # Always retry on EmptyResponseError (transient API issue)
            if isinstance(exception, EmptyResponseError):
                return True
            # Never retry on LLMClientError (permanent failures like context length)
            if isinstance(exception, LLMClientError):
                return False
            # Retry on other exceptions (network errors, timeouts, etc.)
            return True

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=5, max=300),
            retry=retry_if_exception(_should_retry),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _make_api_call() -> str:
            try:
                completion = await self.async_client.chat.completions.create(
                    **api_params,
                )
                return self._extract_text_from_completion(completion)
            except EmptyResponseError:
                # Re-raise to let tenacity retry
                raise
            except LLMClientError:
                # Re-raise to let tenacity skip retry
                raise
            except Exception as e:
                if self._is_context_length_error(e):
                    error_msg = f"Context length exceeded - skipping: {e}"
                    logger.warning(error_msg)
                    raise LLMClientError(error_msg)

                logger.error(
                    f"Error during LLM API call: {e}. Api params: {api_params}",
                )

                # Re-raise the original exception to let tenacity handle retry logic
                raise

        return await _make_api_call()

    def _prepare_api_params(
        self,
        prompt: str,
        model: str,
        temperature: Optional[float],
        top_p: Optional[float],
        system_prompt: Optional[str],
        reasoning_effort: Optional[str] = "low",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare API parameters for the request."""

        if system_prompt:
            messages = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=system_prompt,
                ),
                ChatCompletionUserMessageParam(role="user", content=prompt),
            ]
        else:
            messages = [
                ChatCompletionUserMessageParam(role="user", content=prompt),
            ]

        params: Dict[str, Any] = {"model": model, "messages": messages}

        # For OpenRouter, let the provider handle defaults. For vLLM, include parameters if specified
        if not self.is_openrouter:
            if temperature is not None:
                params["temperature"] = temperature
            if top_p is not None:
                params["top_p"] = top_p
        else:
            params["extra_body"] = {"reasoning": {"effort": reasoning_effort}}

        params.update(kwargs)
        return params

    def _extract_text_from_completion(self, completion: ChatCompletion) -> str:
        """Extract text content from completion response.

        Raises:
            EmptyResponseError: When API returns empty content (retryable)
            LLMClientError: When response has structural issues (not retryable)
        """
        try:
            if not completion.choices:
                # Empty choices could be a transient issue - make it retryable
                logger.warning(
                    "API returned no choices in completion - will retry",
                )
                raise EmptyResponseError("No choices in completion response")

            message = completion.choices[0].message
            content = message.content

            # Mistral fixes as it's responding in reasoning...
            if not content:
                # Check 'reasoning_content'
                if (
                    hasattr(message, "reasoning_content")
                    and message.reasoning_content
                ):
                    content = message.reasoning_content
                # Check 'reasoning'
                elif hasattr(message, "reasoning") and message.reasoning:
                    content = message.reasoning

            # Check finish_reason for issues
            finish_reason = completion.choices[0].finish_reason
            if finish_reason == "content_filter":
                raise LLMClientError(
                    f"Content filtered by API (finish_reason={finish_reason})",
                )

            if not content:
                # Empty content is often a transient issue with OpenRouter/Gemini
                logger.warning(
                    f"API returned empty content (finish_reason={finish_reason}) - will retry (choices={completion.choices})",
                )
                raise EmptyResponseError(
                    f"No content in completion message (finish_reason={finish_reason}) {completion.choices =}",
                )

            if not isinstance(content, str):
                content = str(content)
            return content.strip()
        except (EmptyResponseError, LLMClientError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            raise LLMClientError(f"Error extracting text from completion: {e}")

    def _is_context_length_error(self, error: Exception) -> bool:
        """Check if error is a context length error that should not be retried."""
        error_str = str(error).lower()
        return any(
            phrase in error_str
            for phrase in [
                "context length",
                "context the overflows",
                "maximum context length",
                "token limit",
                "sequence length",
                "context size",
            ]
        )

    def get_cache_key(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Generate cache key for LLM request parameters."""
        params: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
        }

        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p

        for key, value in kwargs.items():
            if value is not None:
                params[key] = value

        content = json.dumps(params, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def load_from_cache(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Load LLM response from cache if available."""
        if not self.cache_dir:
            return None

        try:
            cache_key = self.get_cache_key(
                prompt,
                model,
                system_prompt,
                temperature,
                top_p,
                **kwargs,
            )
            cache_file = self.cache_dir / f"{cache_key}.json"

            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                if (
                    "response" in cache_data
                    and "model" in cache_data
                    and cache_data["model"] == model
                    and "prompt_hash" in cache_data
                ):
                    prompt_hash = hashlib.sha256(
                        prompt.encode("utf-8"),
                    ).hexdigest()
                    if cache_data["prompt_hash"] == prompt_hash:
                        return str(cache_data["response"])

        except Exception as e:
            logger.debug(f"Error loading LLM response from cache: {e}")

        return None

    def save_to_cache(
        self,
        prompt: str,
        response: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Save LLM response to cache."""
        if not self.cache_dir:
            return

        try:
            cache_key = self.get_cache_key(
                prompt,
                model,
                system_prompt,
                temperature,
                top_p,
                **kwargs,
            )
            cache_file = self.cache_dir / f"{cache_key}.json"

            cache_data = {
                "prompt_preview": (
                    prompt[:200] + "..." if len(prompt) > 200 else prompt
                ),
                "prompt_hash": hashlib.sha256(
                    prompt.encode("utf-8"),
                ).hexdigest(),
                "response": response,
                "model": model,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "top_p": top_p,
                "timestamp": time.time(),
                "response_length": len(response),
            }

            for key, value in kwargs.items():
                if value is not None and key not in cache_data:
                    cache_data[f"param_{key}"] = value

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            logger.debug(f"Error saving LLM response to cache: {e}")
