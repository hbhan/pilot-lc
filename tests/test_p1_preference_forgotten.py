"""PROBLEM 1 - "I told it on Monday that Acme's CFO wants a one-page summary. On Thursday it drafted five pages."

ROOT CAUSE   Chat history is per session and finite. After many other sessions the Monday message is gone.
HOW WE FIX   (a) fact_store.py: the preference is saved as a FAISS fact (kind="preference", account="Acme").
             (b) retriever.build_memory: preferences of an in-scope account are ALWAYS included (not
                 similarity-ranked, so they can never be 'outranked').
             (c) prompts.REPLY_SYSTEM rule 3: preferences are binding on drafts.
WHAT THE TEST PROVES  After 12 unrelated sessions and a restart, the new session's prompt contains the
             preference although the verbatim history of that session is EMPTY, so it can only have come
             from FAISS memory.
"""
from tests.helpers import at, make, memory_of

RULES = {
    "one-page": [{"account": "Acme", "person": "Dana Ruiz", "kind": "preference",
                  "text": "Acme's CFO Dana Ruiz wants every proposal as a one-page summary."}],
    "call with": [{"account": "Initech", "kind": "note", "text": "Call with Initech about pricing."}],
}


def test_preference_survives_many_sessions_and_restart(tmp_path):
    a, _ = make(tmp_path, RULES)
    a.start_session("u1", "s01", at(0))
    a.chat("u1", "s01", "Just off a call with Acme. Their CFO, Dana Ruiz, wants every proposal as a one-page summary.", at(0, 1))
    a.end_session("u1", "s01", at(0, 5))

    for i in range(2, 14):                                   # 12 unrelated sessions about other clients
        a.start_session("u1", f"s{i:02d}", at(1, i))
        for k in range(5):
            a.chat("u1", f"s{i:02d}", f"Call with Initech about topic {i}-{k}, long discussion of pricing details.", at(1, i * 10 + k))

    b, llm = make(tmp_path, RULES)                           # RESTART: new objects, same folder
    b.start_session("u1", "s14", at(3))
    b.chat("u1", "s14", "Outline the renewal proposal for Acme's CFO.", at(3, 1))

    assert llm.last_history == []                            # nothing from earlier sessions in the window...
    mem = memory_of(llm)
    assert "one-page summary" in mem and "Dana Ruiz" in mem  # ...yet the preference is in the prompt
    assert "binding" in llm.last_system                      # and the rule that makes it binding is too
