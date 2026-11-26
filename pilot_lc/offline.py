"""Offline TEST DOUBLES so every problem can be demonstrated and tested with NO API key and NO network.

They replace exactly the three things that normally call OpenAI:
    HashingEmbeddings   <-> OpenAIEmbeddings   (text -> vector)
    scripted_extractor  <-> extractor LLM      (message -> JSON facts)
    RecordingChatModel  <-> ChatOpenAI         (prompt -> reply; also RECORDS the prompt for assertions)

IMPORTANT honesty note: with these doubles a test proves the PIPELINE (what reaches the model, in what shape,
within what budget). It does not prove how a real LLM will phrase an answer. Use scripts/walkthrough.py --live
for that.
"""
from __future__ import annotations

import json
import re
import zlib
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field


class HashingEmbeddings(Embeddings):
    """Bag-of-words hashed into 256 dims. Texts sharing words get nearby vectors, which is all FAISS
    needs for our tests. (Real OpenAI embeddings also capture synonyms; this one does not - see GUIDE.)"""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.DIM, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if len(tok) > 2:
                v[zlib.crc32(tok.encode()) % self.DIM] += 1.0   # crc32 = stable across runs (unlike hash())
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def scripted_extractor(rules: dict[str, list[dict]] | None = None) -> RunnableLambda:
    """A fake extraction LLM. `rules` maps a trigger phrase -> the facts a good LLM WOULD extract.

    Example: {"one-page": [{"account": "Acme", "kind": "preference", "text": "Acme's CFO wants one-page summaries"}]}
    Making the 'expected LLM output' explicit in each test is the point: the tests then show exactly which
    part of OUR code (storage, scoping, budget) is being verified.
    """
    rules = {k.lower(): v for k, v in (rules or {}).items()}

    def run(prompt_value) -> str:
        text = prompt_value.to_string()
        message = text.split("MESSAGE:", 1)[1].lower()           # only look at the user's message part
        facts: list[dict] = []
        for trigger, produced in rules.items():
            if trigger in message:
                facts += produced
        return json.dumps({"facts": facts})                      # same JSON shape the real LLM must return

    return RunnableLambda(run)


class RecordingChatModel(BaseChatModel):
    """A fake chat model that answers 'ok' and records every prompt it was sent."""

    calls: list = Field(default_factory=list)   # each entry = the list of messages of one call

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.calls.append(messages)
        # Report input tokens the way OpenAI does (usage_metadata), using the same approximate counter.
        tokens = count_tokens_approximately(messages)
        msg = AIMessage(content="ok", usage_metadata={"input_tokens": tokens, "output_tokens": 1,
                                                      "total_tokens": tokens + 1})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # ---- convenience accessors for tests -------------------------------------------------------------
    @property
    def last_system(self) -> str:
        return self.calls[-1][0].content            # first message of the prompt = system (rules + MEMORY)

    @property
    def last_history(self) -> list:
        return self.calls[-1][1:-1]                 # messages between system and the final question
