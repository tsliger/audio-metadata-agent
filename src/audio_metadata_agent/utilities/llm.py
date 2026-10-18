import os

from langchain_openai.chat_models.base import ChatOpenAI
from pydantic.types import SecretStr


def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY cannot be null or empty")

    if not model_name:
        raise ValueError("OPENROUTER_MODEL cannot be null or empty")

    llm = ChatOpenAI(
        api_key=SecretStr(api_key),
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        temperature=0.0,
    )

    return llm
