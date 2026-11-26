"""Factories for the three OpenAI objects. Isolated here so (a) swapping provider touches one file and
(b) tests can inject fakes without any network or key."""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import Settings


def _key(settings: Settings) -> str:
    """Read the API key from the environment variable NAMED in settings (fails loudly if missing)."""
    return os.environ[settings.api_key_env]


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """The model that writes user-facing replies."""
    return ChatOpenAI(model=settings.chat_model, temperature=settings.temperature,
                      max_tokens=settings.max_reply_tokens, api_key=_key(settings))


def build_extractor_model(settings: Settings):
    """The model that extracts facts. `response_format=json_object` forces valid JSON, which makes the
    Pydantic parser in extractor.py far less likely to fail (the prompt already contains the word JSON)."""
    llm = ChatOpenAI(model=settings.extract_model, temperature=0, api_key=_key(settings))
    return llm.bind(response_format={"type": "json_object"})


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Text -> vector, used by FAISS both when saving facts and when searching them."""
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=_key(settings))
