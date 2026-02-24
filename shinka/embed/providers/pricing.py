# Available embedding models and pricing - loaded from pricing.csv as DataFrame
# OpenAI: https://platform.openai.com/docs/pricing
# Gemini: https://ai.google.dev/gemini-api/docs/pricing

import pandas as pd
from pathlib import Path
from typing import Optional

# Load pricing data from CSV
_pricing_csv_path = Path(__file__).parent / "pricing.csv"
# Utility constant
M = 1_000_000


def _load_pricing_dataframe() -> pd.DataFrame:
    """Load pricing data from CSV file as a pandas DataFrame."""
    df = pd.read_csv(_pricing_csv_path)

    # Strip whitespace from string columns only
    for col in df.columns:
        if df[col].dtype == "object":  # Only strip string columns
            df[col] = df[col].str.strip()

    # Strip column names
    df.columns = df.columns.str.strip()

    # Convert price column to numeric (handling N/A as 0)
    df["input_price"] = pd.to_numeric(
        df["input_price"].replace("N/A", "0"), errors="coerce"
    )

    # Convert prices from per-1M-tokens to per-token
    df["input_price"] = df["input_price"] / M

    # Set index to model_name for fast lookups
    df = df.set_index("model_name")

    return df


# Load pricing dataframe
_PRICING_DF = _load_pricing_dataframe()


def _parse_openrouter_model(model_name: str) -> Optional[str]:
    """Extract the base model name from an OpenRouter 'company/model' format.

    Returns the model suffix (after '/') if it's an OpenRouter pattern,
    otherwise returns None.

    Example: 'openai/text-embedding-3-small' -> 'text-embedding-3-small'
    """
    if "/" in model_name and not model_name.startswith("azure-"):
        return model_name.split("/", 1)[1]
    return None


def _resolve_model_name(model_name: str) -> Optional[str]:
    """Resolve a model name to its CSV entry.

    For direct entries, returns the model_name if found.
    For OpenRouter models (company/model), returns the base model if found in CSV.
    Returns None if no matching entry exists.
    """
    # Direct lookup
    if model_name in _PRICING_DF.index:
        return model_name
    # OpenRouter pattern: check if base model exists
    base_model = _parse_openrouter_model(model_name)
    if base_model and base_model in _PRICING_DF.index:
        return base_model
    return None


def get_model_price(model_name: str) -> float:
    """Get the input price per token for a model.

    Returns the input price (embeddings only have input costs).
    Supports OpenRouter 'company/model' format by looking up the base model.
    """
    resolved = _resolve_model_name(model_name)
    if resolved is None:
        raise ValueError(f"Embedding model {model_name} not found in pricing data")
    return _PRICING_DF.loc[resolved, "input_price"]


def model_exists(model_name: str) -> bool:
    """Check if an embedding model exists in pricing data.

    Supports both direct model names and OpenRouter 'company/model' format
    where the base model must exist in the CSV.
    """
    return _resolve_model_name(model_name) is not None


def get_provider(model_name: str) -> Optional[str]:
    """Get the provider for a given embedding model.

    For OpenRouter models (company/model format), returns 'openrouter'
    if the base model exists in CSV, regardless of the CSV's provider field.
    """
    # Check if it's an OpenRouter model with valid base
    base_model = _parse_openrouter_model(model_name)
    if base_model and base_model in _PRICING_DF.index:
        return "openrouter"
    # Direct lookup
    if model_name in _PRICING_DF.index:
        return _PRICING_DF.loc[model_name, "provider"]
    return None


def get_all_models() -> list:
    """Get list of all available embedding model names."""
    return _PRICING_DF.index.tolist()


def get_models_by_provider(provider: str) -> list:
    """Get list of embedding models for a given provider."""
    return _PRICING_DF[_PRICING_DF["provider"] == provider].index.tolist()
