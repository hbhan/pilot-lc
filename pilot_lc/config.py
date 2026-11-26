"""All tunable knobs live in ONE place so a reader can see every trade-off at a glance.

Each field documents: what it does, why the default, and what you give up by changing it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)  # frozen = settings cannot be mutated by accident at runtime
class Settings:
    # ------------------------------------------------------------------ models
    # Model that writes the user-facing reply.
    # Trade-off: a bigger model follows "apply the client's preference" rules more reliably, but costs more
    # per reply. Replies are the hot path, so this is the knob to turn up if quality matters most.
    chat_model: str = "gpt-4o-mini"

    # Model that reads each user message and turns it into facts ("extraction").
    # Trade-off: extraction is a simple, structured task -> a SMALL model is enough and keeps the
    # one-extra-call-per-message cost low. A weak extractor, however, silently writes wrong memory.
    extract_model: str = "gpt-4o-mini"

    # Model that turns text into vectors for FAISS similarity search.
    # Trade-off: "-small" is ~6x cheaper than "-large"; retrieval here is scoped to one account, so
    # the extra precision of a larger model rarely pays for itself.
    embedding_model: str = "text-embedding-3-small"

    # NAME of the environment variable holding the key (never the key itself: keys stay out of code/logs).
    api_key_env: str = "OPENAI_API_KEY"

    # 0.0 = most repeatable answers. Memory-grounded answers should not be "creative" about facts.
    temperature: float = 0.0

    # Cap on reply length. Only protects cost/latency; it does not count toward the INPUT limit.
    max_reply_tokens: int = 700

    # ---------------------------------------------------------- token budget
    # The contract: every reply call may send at most this many INPUT tokens.
    hard_limit_tokens: int = 8000

    # What we PLAN to. Our token counter is approximate (about 4 chars/token) so we multiply by
    # `safety_factor` and plan to land under 7000, leaving ~1000 tokens of headroom below the hard limit.
    # Trade-off: more headroom = safer but less room for memory/history.
    target_limit_tokens: int = 7000
    safety_factor: float = 1.3

    # Max tokens of the CURRENT session's recent turns sent verbatim (LangChain `trim_messages`).
    # Trade-off: bigger = better short-term coherence, but pushes out memory space and raises cost on every
    # call. Long-term knowledge comes from FAISS facts, so this can stay small.
    history_tokens: int = 2000

    # Max tokens of the user's CURRENT message sent to the reply model. Longer text (e.g. pasted chats)
    # is still fully ingested into memory; only a head+tail excerpt goes into the reply prompt.
    message_tokens: int = 1500

    # ---------------------------------------------------------------- memory
    # How many *relevant extra* facts (notes/events) to retrieve per account via similarity search.
    # Preferences/contacts/status are ALWAYS included separately (see `core_facts`).
    # Trade-off: higher = better recall of old details, but more tokens and more chance of distracting noise.
    relevant_facts: int = 6

    # Max "always-on" facts (preferences, contacts, status, commitments) shown per account, newest first.
    core_facts: int = 15

    # If the user names several accounts in one message, show at most this many (token control).
    max_accounts_in_scope: int = 3

    # Pasted text is cut into chunks of this many characters (~2000 tokens) before extraction so each
    # extraction call stays small and accurate. Trade-off: smaller chunks = more calls but fewer misses.
    chunk_chars: int = 8000

    # Messages shorter than this ("ok thanks") are not sent to the extractor: saves a call, loses nothing.
    min_chars_to_extract: int = 20

    @classmethod
    def from_dict(cls, config: dict | None) -> "Settings":
        """Build Settings from the contract's `config` dict.

        The contract guarantees {"model": ..., "api_key_env": ...}. We map "model" -> chat_model and accept
        any other key that matches a field name; unknown keys are ignored so the harness can pass extras.
        """
        config = dict(config or {})
        if "model" in config:                       # contract key name -> our field name
            config["chat_model"] = config.pop("model")
        allowed = {f.name for f in fields(cls)}    # names of all settings fields
        return cls(**{k: v for k, v in config.items() if k in allowed})
