"""Pilot, LangChain edition. This file is the ORCHESTRATOR: it wires the pieces together and nothing else.

Read it top to bottom; `chat()` is the whole story, in four numbered steps:

    1. LEARN    save what the user just said into FAISS memory            (Problems 1, 2, 3, 4)
    2. SCOPE    work out which client(s) the message is about, fetch ONLY their facts  (Problem 5)
    3. FIT      shrink memory until the prompt is safely under the token limit        (hard 8,000 limit)
    4. ANSWER   one LangChain call: system rules + memory + recent turns + question   (Problems 4, 6)

Why this order?  LEARN comes first so facts are saved even if the reply call later fails (write-ahead).
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from operator import itemgetter
from pathlib import Path

from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .extractor import Extractor
from .fact_store import FactStore
from .models import build_chat_model, build_embeddings, build_extractor_model
from .prompts import REPLY_PROMPT, REPLY_SYSTEM
from .retriever import build_memory


# LangChain 1.x marks RunnableWithMessageHistory as "legacy" (successor: LangGraph checkpointers). We use it on
# purpose: it is the classic, easy-to-explain "message history" pattern. Silence only that one warning.
warnings.filterwarnings("ignore", message="RunnableWithMessageHistory is deprecated")


@dataclass
class Reply:
    """What the harness expects back from chat()."""
    text: str            # what the user sees
    input_tokens: int    # input tokens of the call that produced this reply


def _safe(name: str) -> str:
    """Make a user/session id safe to use as a file name (blocks path tricks like '../../etc')."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _clip_middle(text: str, max_tokens: int) -> str:
    """Keep the start and end of an over-long text (~4 chars per token). The start usually states the
    intent ('here are my old notes') and the end holds the latest content; the middle is what we drop."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    head, tail = int(max_chars * 0.6), int(max_chars * 0.4)
    return text[:head] + "\n[... long message shortened; full text was saved to memory ...]\n" + text[-tail:]


class Assistant:
    def __init__(self, data_dir: str, config: dict | None = None, *,
                 llm=None, extractor_llm=None, embeddings=None) -> None:
        """Load or create all state under `data_dir` (contract rule).

        The keyword-only `llm`, `extractor_llm`, `embeddings` arguments are OPTIONAL dependency injection.
        Production passes nothing (OpenAI objects are built from config); tests pass offline fakes.
        """
        self.cfg = Settings.from_dict(config)                         # all knobs, with documented defaults
        self.root = Path(data_dir)                                    # everything persists under this folder
        (self.root / "users").mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings or build_embeddings(self.cfg)    # text -> vectors for FAISS
        self.llm = llm or build_chat_model(self.cfg)                  # writes user-facing replies
        self.extractor = Extractor(extractor_llm or build_extractor_model(self.cfg))  # message -> facts
        # Splits long text on paragraph/sentence boundaries. For normal short messages it returns ONE chunk,
        # so we can use it unconditionally (no special case for "pasted history").
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.cfg.chunk_chars, chunk_overlap=200)
        self._stores: dict[str, FactStore] = {}                       # cache: one FAISS store per user
        self._chain = self._build_chain()                             # the LangChain pipeline (see below)
        # Debug/explainability hooks: what the model saw on the last call (used by the walkthrough script).
        self.last_memory = ""
        self.last_input_tokens = 0

    # ------------------------------------------------------------------------------- LangChain wiring
    def _build_chain(self) -> RunnableWithMessageHistory:
        """Build:   history -> trim -> prompt -> llm, wrapped so history is loaded/saved automatically.

        * RunnableWithMessageHistory: LangChain's standard way to give a chain per-session chat history.
          It loads the session's messages before the call and appends the new human+AI messages after.
        * trim_messages: keeps only the most recent turns that fit `history_tokens`. This bounds the
          verbatim window so a long day of calls cannot overflow the context (Problem 4). Anything trimmed
          away was already saved as facts in step 1, so nothing is lost.
        """
        trimmer = trim_messages(
            max_tokens=self.cfg.history_tokens,
            token_counter=count_tokens_approximately,   # fast, offline estimate (no tokenizer download)
            strategy="last",                            # keep the NEWEST messages
            start_on="human",                           # never start the window on an orphan AI reply
            include_system=False,
        )
        core = (
            RunnablePassthrough.assign(history=itemgetter("history") | trimmer)  # replace history by trimmed copy
            | REPLY_PROMPT
            | self.llm
        )
        return RunnableWithMessageHistory(
            core, self._history,
            input_messages_key="question",      # which input key holds the new human message
            history_messages_key="history",     # which prompt slot receives the past messages
        )

    def _history(self, session_id: str) -> FileChatMessageHistory:
        """Return the (file-backed) chat history of ONE session. File-backed = survives a process restart.
        `session_id` arrives as 'user::session' so the same session name used by two users never collides."""
        user, session = session_id.split("::", 1)
        folder = self.root / "users" / _safe(user) / "history"
        folder.mkdir(parents=True, exist_ok=True)
        return FileChatMessageHistory(str(folder / f"{_safe(session)}.json"))

    def _store(self, user_id: str) -> FactStore:
        """Get (or lazily load from disk) this user's FAISS fact store."""
        if user_id not in self._stores:
            self._stores[user_id] = FactStore(self.root / "users" / _safe(user_id) / "facts", self.embeddings)
        return self._stores[user_id]

    # ----------------------------------------------------------------------------------- public contract
    def start_session(self, user_id: str, session_id: str, ts: datetime) -> None:
        """Nothing to do: history files are created on first message, memory is loaded on demand.
        Kept (a) because the contract requires it and (b) because it shows what we DON'T need:
        no 'recover unfinished session' logic, since facts are saved on every message (see chat step 1)."""

    def end_session(self, user_id: str, session_id: str, ts: datetime) -> None:
        """Also nothing: users close tabs and the contract says this may never be called.
        Design consequence: NOTHING important may depend on this method. Learning happens in chat()."""

    def chat(self, user_id: str, session_id: str, message: str, ts: datetime) -> Reply:
        store = self._store(user_id)

        # STEP 1 - LEARN ----------------------------------------------------------------------------
        self._learn(store, message, ts)

        # STEP 2 - SCOPE + FIT ------------------------------------------------------------------------
        question = _clip_middle(message, self.cfg.message_tokens)   # what the reply model will see
        today = ts.strftime("%A %Y-%m-%d")                          # time comes from `ts`, never the clock
        memory, estimated_tokens = self._fit_memory(store, question, today)

        # STEP 3 - ANSWER -----------------------------------------------------------------------------
        sid = f"{user_id}::{session_id}"
        try:
            result = self._chain.invoke(
                {"question": question, "memory": memory, "today": today},
                config={"configurable": {"session_id": sid}},
            )
            text = result.content
            # Prefer the provider's own count (OpenAI returns it in usage_metadata); else our estimate.
            usage = getattr(result, "usage_metadata", None) or {}
            input_tokens = usage.get("input_tokens") or estimated_tokens
        except Exception as exc:   # provider outage: stay usable; the user's facts are already saved (step 1)
            text = f"Sorry, I could not reach the model just now ({type(exc).__name__}). Your message is saved."
            input_tokens = estimated_tokens

        self.last_memory, self.last_input_tokens = memory, input_tokens
        return Reply(text=text, input_tokens=input_tokens)

    # ------------------------------------------------------------------------------------------ internals
    def _learn(self, store: FactStore, message: str, ts: datetime) -> None:
        """Extract facts from the user's message and save them to FAISS.

        Runs on EVERY user message (simple + no loss, at the price of one small-model call per message).
        Long pasted text is split into chunks first, so a 50-page paste becomes many small extractions
        instead of one overloaded call (Problem 3). The reply call never sees the whole paste.
        """
        if len(message.strip()) < self.cfg.min_chars_to_extract:    # "ok thanks" -> nothing to learn
            return
        for chunk in self.splitter.split_text(message):
            for fact in self.extractor.extract(chunk, ts, store.accounts()):
                store.add(fact, ts)

    def _fit_memory(self, store: FactStore, question: str, today: str) -> tuple[str, int]:
        """Build the memory block, shrinking it until the whole prompt is safely under the limit.

        Budget = system rules + memory + history window + question.  We budget the history at its MAXIMUM
        (history_tokens) even if today's history is shorter: worst-case planning means the guarantee holds
        however long the session gets. The estimate is multiplied by safety_factor because our counter is
        approximate; the result must stay under target_limit_tokens (< the hard 8,000 limit).
        """
        n_relevant, n_core = self.cfg.relevant_facts, self.cfg.core_facts
        while True:
            memory = build_memory(store, question, n_relevant, n_core, self.cfg.max_accounts_in_scope)
            system_text = REPLY_SYSTEM.format(today=today, memory=memory)
            raw = count_tokens_approximately([SystemMessage(system_text), HumanMessage(question)])
            estimate = int((raw + self.cfg.history_tokens) * self.cfg.safety_factor)
            if estimate <= self.cfg.target_limit_tokens:
                return memory, estimate
            if n_relevant == 0 and n_core <= 1:     # cannot shrink structurally any more -> hard truncate
                # Last resort (only if the system rules + question alone are huge): cut the memory text itself.
                allowed_tokens = int(self.cfg.target_limit_tokens / self.cfg.safety_factor) - self.cfg.history_tokens
                return memory[: max(200, allowed_tokens * 3)], self.cfg.target_limit_tokens
            n_relevant //= 2                        # shrink the cheap, low-value part first
            n_core = max(1, n_core // 2)            # then the always-on part, newest facts kept
