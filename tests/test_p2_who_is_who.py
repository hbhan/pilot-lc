"""PROBLEM 2 - "Every morning I have to re-explain who is who on my accounts."

ROOT CAUSE   No durable entity memory: people/roles live only in yesterday's chat.
HOW WE FIX   Contacts are saved once as facts (kind="contact", with person + account in metadata) and the
             per-user `accounts.json` records client names. build_memory shows them whenever the client is named,
             and (soft scope) when only a person is named.
WHAT THE TEST PROVES  Contacts told on different days are all available after a restart, with clear labels.
"""
from tests.helpers import at, make, memory_of

RULES = {
    "dana ruiz": [{"account": "Acme", "person": "Dana Ruiz", "kind": "contact", "key": "contact.cfo",
                   "text": "Dana Ruiz is the CFO at Acme."}],
    "tom hale": [{"account": "Globex", "person": "Tom Hale", "kind": "contact", "key": "contact.procurement",
                  "text": "Tom Hale is the procurement lead at Globex."}],
}


def test_contacts_persist_and_are_labelled(tmp_path):
    a, _ = make(tmp_path, RULES)
    a.chat("u1", "mon", "Met Dana Ruiz today, she is the CFO at Acme.", at(0))
    a.chat("u1", "tue", "Tom Hale is our procurement lead contact at Globex.", at(1))

    b, llm = make(tmp_path, RULES)                           # next morning, fresh process
    b.chat("u1", "wed", "Who do I talk to at Acme?", at(2))
    mem = memory_of(llm)
    assert "Dana Ruiz is the CFO at Acme" in mem
    assert "Accounts on file: Acme, Globex" in mem           # the 'roster' of clients is always present

    b.chat("u1", "wed", "Remind me who Tom Hale is", at(2, 5))  # person named, client not named
    assert "[Globex] (contact) Tom Hale is the procurement lead" in memory_of(llm)
