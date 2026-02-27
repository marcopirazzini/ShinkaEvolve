from dataclasses import dataclass
from typing import Any, Tuple
import os
import openai
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from .providers.pricing import get_provider

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

TIMEOUT = 600

_OPENROUTER_PREFIX = "openrouter/"
_AZURE_PREFIX = "azure-"


@dataclass(frozen=True)
class ResolvedEmbedModel:
    original_model_name: str
    api_model_name: str
    provider: str


def resolve_embed_model(model_name: str) -> ResolvedEmbedModel:
    """Resolve embedding model backend with fallback pattern matching.

    Resolution order:
    1. Check pricing.csv for explicit entries
    2. Check for 'openrouter/' prefix
    3. Check for 'azure-' prefix
    """
    # First check pricing.csv
    provider = get_provider(model_name)
    if provider is not None:
        api_model_name = model_name
        # Strip azure- prefix for Azure models from pricing.csv
        if provider == "azure":
            api_model_name = model_name.split(_AZURE_PREFIX, 1)[-1]
        return ResolvedEmbedModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider=provider,
        )

    # Fallback: check for openrouter/ prefix
    if model_name.startswith(_OPENROUTER_PREFIX):
        api_model_name = model_name.split(_OPENROUTER_PREFIX, 1)[-1]
        if not api_model_name:
            raise ValueError("OpenRouter model name is missing after 'openrouter/'.")
        return ResolvedEmbedModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="openrouter",
        )

    # Fallback: check for azure- prefix
    if model_name.startswith(_AZURE_PREFIX):
        api_model_name = model_name.split(_AZURE_PREFIX, 1)[-1]
        if not api_model_name:
            raise ValueError("Azure model name is missing after 'azure-' prefix.")
        return ResolvedEmbedModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="azure",
        )

    raise ValueError(
        f"Embedding model '{model_name}' is not supported. "
        "Use a known pricing.csv model, 'openrouter/<model>', or 'azure-<model>'."
    )


def get_client_embed(model_name: str) -> Tuple[Any, str]:
    """Get the client and model for the given embedding model name.

    Args:
        model_name (str): The name of the embedding model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        Tuple[Any, str]: The client and model name for the given model.
    """
    resolved = resolve_embed_model(model_name)
    provider = resolved.provider
    api_model_name = resolved.api_model_name

    if provider == "openai":
        client = openai.OpenAI(timeout=TIMEOUT)
    elif provider == "azure":
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
            timeout=TIMEOUT,
        )
    elif provider == "google":
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    elif provider == "openrouter":
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, api_model_name


def get_async_client_embed(model_name: str) -> Tuple[Any, str]:
    """Get the async client and model for the given embedding model name.

    Args:
        model_name (str): The name of the embedding model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        Tuple[Any, str]: The async client and model name for the given model.
    """
    resolved = resolve_embed_model(model_name)
    provider = resolved.provider
    api_model_name = resolved.api_model_name

    if provider == "openai":
        client = openai.AsyncOpenAI()
    elif provider == "azure":
        client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
        )
    elif provider == "google":
        # Gemini doesn't have async client yet, will use thread pool in embedding.py
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    elif provider == "openrouter":
        client = openai.AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, api_model_name
