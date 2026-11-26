"""Decides WHICH memory goes into the prompt and formats it. Two jobs:

1. Scope:   which client(s) is the message about?      -> `find_accounts`
2. Block:   build the MEMORY text for the reply prompt -> `build_memory`
"""
from __future__ import annotations

import re

from .fact_store import FactStore


def find_accounts(message: str, known_accounts: list[str]) -> list[str]:
    """Client names that appear in the message, as whole words, case-insensitive.

    Whole-word matching matters: 'Acme' must not match inside 'Acmeville'.
    Deliberately simple (no LLM): it is fast, free, deterministic and easy to explain.
    Known limit: nicknames ('the Acme guys' works, 'AC') do not - see GUIDE trade-offs.
    """
    found = []
    for name in known_accounts:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", message, flags=re.I):
            found.append(name)
    return found


def _line(doc) -> str:
    """Render one fact as a prompt line: label + text + date logged. The date lets the model judge freshness."""
    acct = doc.metadata.get("account") or "general"
    return f"- [{acct}] ({doc.metadata['kind']}) {doc.page_content} (logged {doc.metadata['ts'][:10]})"


def build_memory(store: FactStore, message: str, n_relevant: int, n_core: int, max_accounts: int) -> str:
    """Assemble the MEMORY block. Two modes, chosen by whether the user NAMED a client:

    * Named client  -> HARD scope: only that client's facts (Problem 5). Preferences/contacts first.
    * No client     -> SOFT scope: a labelled similarity search across all clients (e.g. "what was the
                       deadline from my first call?" - Problem 4). Labels let the model see whose fact it is.
    """
    accounts = store.accounts()
    lines = [f"Accounts on file: {', '.join(accounts) if accounts else '(none yet)'}"]

    own = store.user_level(n_core)                                   # the user's own working style
    if own:
        lines.append("== The user's own standing preferences (apply to every client) ==")
        lines += [_line(d) for d in own]

    named = find_accounts(message, accounts)[:max_accounts]
    if named:
        for acct in named:
            core, extra = store.for_account(acct, message, n_relevant, n_core)
            lines.append(f"== {acct} ==")
            lines += [_line(d) for d in core]                        # preferences, contacts, status, commitments
            lines += [_line(d) for d in extra]                       # similar notes/events
            if not core and not extra:
                lines.append(f"- No facts on record for {acct}.")
    else:
        hits = store.search_all(message, n_relevant)
        if hits:
            lines.append("== Possibly relevant facts (labelled by client; use only if the user means that client) ==")
            lines += [_line(d) for d in hits]

    # Explicit boundary: tells the model that absence from this block means "not on record" [Problem 6].
    lines.append("END OF MEMORY. Anything not listed above is NOT on record.")
    return "\n".join(lines)
