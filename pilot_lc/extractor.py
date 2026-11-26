"""Turns one user message into a list of `Fact`s using an LLM, via a standard LangChain chain:

        prompt  |  llm  |  PydanticOutputParser

`PydanticOutputParser` validates the LLM output against the `FactList` schema, so malformed output is
caught here instead of corrupting memory.
"""
from __future__ import annotations

from datetime import datetime

from langchain_core.output_parsers import PydanticOutputParser

from .fact_store import Fact, FactList
from .prompts import EXTRACT_PROMPT
from .retriever import find_accounts


class Extractor:
    def __init__(self, llm) -> None:
        parser = PydanticOutputParser(pydantic_object=FactList)
        # .partial() pre-fills {format_instructions} so callers only supply per-message values.
        prompt = EXTRACT_PROMPT.partial(format_instructions=parser.get_format_instructions())
        self.chain = prompt | llm | parser

    def extract(self, message: str, ts: datetime, known_accounts: list[str]) -> list[Fact]:
        """Return facts found in `message`. NEVER raises and never drops the user's words:
        if the LLM call or parsing fails we keep the raw text as a 'note' fact instead (degrade, don't lose).
        """
        try:
            result = self.chain.invoke({
                "accounts": ", ".join(known_accounts) or "(none yet)",
                "date": ts.strftime("%A %Y-%m-%d"),     # weekday lets the LLM resolve "on Thursday"
                "message": message,
            })
            return result.facts
        except Exception:
            named = find_accounts(message, known_accounts)   # attach to a client if one is named
            return [Fact(account=named[0] if named else None, kind="note", text=message[:300])]
