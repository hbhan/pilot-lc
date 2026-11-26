"""The memory itself: a per-user FAISS vector store of small, typed, dated FACTS.

WHY FACTS (not raw chat chunks)?  A raw chunk says "Dana wants one page, also the weather was bad".
A fact says "Dana Ruiz (CFO, Acme) wants proposals as a one-page summary" and carries metadata
(account, kind, key, date). Metadata is what lets us (a) restrict a search to ONE client account
[Problem 5], (b) always include preferences [Problem 1], (c) replace outdated facts [corrections].

WHY ONE INDEX PER USER?  Isolation by construction: a user's search physically cannot return another
user's data because it is a different file on disk. ("Two users never share information.")

Layout on disk:  <data_dir>/users/<user_id>/facts/{index.faiss, index.pkl, accounts.json}
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# Kinds that are ALWAYS shown for an in-scope account (small, high-value, rarely numerous).
# "event" and "note" are only shown when similar to the question (they can be hundreds).
CORE_KINDS = {"preference", "contact", "status", "commitment"}


class Fact(BaseModel):
    """One unit of memory. This same class is the schema the extraction LLM must produce."""
    account: Optional[str] = Field(None, description="Client name, or null for the user's own working style")
    person: Optional[str] = Field(None, description="Full name of the person the fact is about, if any")
    kind: Literal["preference", "contact", "status", "commitment", "event", "note"] = "note"
    key: Optional[str] = Field(None, description="Stable label if this REPLACES an older fact, e.g. contact.cfo")
    text: str = Field(..., description="One self-contained sentence naming the client and person")


class FactList(BaseModel):
    """Wrapper so the LLM returns a JSON object {"facts": [...]} (json_object mode requires an object)."""
    facts: list[Fact] = Field(default_factory=list)


class FactStore:
    def __init__(self, folder: Path, embeddings) -> None:
        self.folder = folder                              # where this user's index lives
        self.embeddings = embeddings                      # text->vector function (OpenAI or a test double)
        self.index: FAISS | None = None                   # None until the first fact is saved
        self._accounts_file = folder / "accounts.json"    # small list of client names for fast name matching
        if (folder / "index.faiss").exists():             # restart-safety: reload what a previous run saved
            # allow_dangerous_deserialization: FAISS metadata is a pickle. Safe here because WE wrote the
            # file in our own data dir; never enable this for files from an untrusted source.
            self.index = FAISS.load_local(str(folder), embeddings, allow_dangerous_deserialization=True)

    # ------------------------------------------------------------------------------------------ helpers
    def accounts(self) -> list[str]:
        """Client names this user has told us about (used to detect 'which client is this message about?')."""
        return json.loads(self._accounts_file.read_text()) if self._accounts_file.exists() else []

    def _size(self) -> int:
        """Number of facts stored. We search with k=size so FAISS metadata filters see EVERY fact
        (FAISS filters AFTER nearest-neighbour search, so a small fetch_k could silently miss matches)."""
        return self.index.index.ntotal if self.index else 0

    def _canonical(self, name: str) -> str:
        """'acme', 'ACME ' and 'Acme' are one client. Simple case/space-insensitive match (see GUIDE trade-offs)."""
        for known in self.accounts():
            if known.lower() == name.strip().lower():
                return known
        return name.strip()

    # ---------------------------------------------------------------------------------------------- write
    def add(self, fact: Fact, ts: datetime) -> None:
        """Save one fact. `ts` is the time the USER said it (from the harness), not the wall clock."""
        self.folder.mkdir(parents=True, exist_ok=True)
        account = self._canonical(fact.account) if fact.account else None

        # Duplicate guard: the user may repeat themselves ("as I said, the CFO wants one page"). Storing the same
        # sentence twice wastes prompt tokens, so skip if this client already has the identical fact.
        if self._already_known(account, fact.text.strip()[:300]):
            return

        # Correction handling: a new fact with the same (account, key) REPLACES the old one,
        # e.g. "Marcus Lee is now Acme's CFO" deletes "Dana Ruiz is Acme's CFO".
        if account and fact.key:
            self._delete_same_key(account, fact.key)

        fact_id = str(uuid.uuid4())                       # stable id so we can delete this fact later
        doc = Document(
            page_content=fact.text.strip()[:300],         # cap length: bounds tokens per fact in every prompt
            metadata={"fid": fact_id, "account": account, "person": fact.person, "kind": fact.kind,
                      "key": fact.key, "ts": ts.isoformat()},
        )
        if self.index is None:                            # FAISS cannot be created empty -> create on first fact
            self.index = FAISS.from_documents([doc], self.embeddings, ids=[fact_id])
        else:
            self.index.add_documents([doc], ids=[fact_id])
        self.index.save_local(str(self.folder))           # persist immediately: a crash cannot lose the fact

        if account and account not in self.accounts():    # remember the client name
            self._accounts_file.write_text(json.dumps(self.accounts() + [account]))

    def _already_known(self, account: str | None, text: str) -> bool:
        n = self._size()
        if not n:
            return False
        same = self.index.similarity_search(text, k=1, fetch_k=n, filter={"account": account})
        return bool(same) and same[0].page_content == text

    def _delete_same_key(self, account: str, key: str) -> None:
        n = self._size()
        if not n:
            return
        old = self.index.similarity_search(key, k=n, fetch_k=n, filter={"account": account, "key": key})
        if old:
            self.index.delete([d.metadata["fid"] for d in old])

    # ----------------------------------------------------------------------------------------------- read
    def for_account(self, account: str, query: str, n_relevant: int, n_core: int):
        """Facts for ONE client: all core facts (newest first) + the `n_relevant` most similar other facts.

        This is the heart of Problem 5: the filter {"account": account} means facts of other clients are
        never even candidates, so they cannot leak into the prompt.
        """
        n = self._size()
        if not n:
            return [], []
        # One call: every fact of this account, ordered by similarity to the user's question.
        docs = self.index.similarity_search(query, k=n, fetch_k=n, filter={"account": account})
        core = [d for d in docs if d.metadata["kind"] in CORE_KINDS]
        core.sort(key=lambda d: d.metadata["ts"], reverse=True)       # newest first
        rest = [d for d in docs if d.metadata["kind"] not in CORE_KINDS]  # already similarity-ordered
        return core[:n_core], rest[:n_relevant]

    def user_level(self, n: int) -> list[Document]:
        """Facts about the user's OWN style (account=None), e.g. 'I like bullet points'."""
        total = self._size()
        if not total:
            return []
        docs = self.index.similarity_search("preference", k=total, fetch_k=total, filter={"account": None})
        return sorted(docs, key=lambda d: d.metadata["ts"], reverse=True)[:n]

    def search_all(self, query: str, k: int) -> list[Document]:
        """Similarity search across ALL of this user's clients. Used only when the message names no client."""
        return self.index.similarity_search(query, k=k) if self.index else []
