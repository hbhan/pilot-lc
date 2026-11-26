"""PROBLEM 6 - "I asked when the Globex pilot started. It gave a confident date I never mentioned."

ROOT CAUSE   Nothing tells the model what it does NOT know, so it fills the gap with a plausible date.
HOW WE FIX   Three layers (cheap -> strong):
             1. retriever.build_memory ends with 'END OF MEMORY. Anything not listed above is NOT on record.'
             2. prompts.REPLY_SYSTEM rule 4: say 'no record' and ask - NEVER guess.
             3. Memory contains only facts the USER stated (extractor prompt: 'never infer, never add dates'),
                and the assistant's own text is never saved as a fact, so it cannot launder its guesses.
WHAT THE TEST PROVES  The prompt for a Globex question contains Globex's real facts, NO start-date fact, the
             explicit boundary line, and the 'never guess' rule. WHETHER the live LLM obeys is only measurable
             with a real model (scripts/walkthrough.py --live); offline we can verify the inputs, not the output.
"""
from tests.helpers import at, make, memory_of

RULES = {"tom hale": [{"account": "Globex", "person": "Tom Hale", "kind": "contact",
                       "text": "Tom Hale is the procurement lead at Globex and wants invoices in EUR."}]}


def test_prompt_makes_absence_explicit(tmp_path):
    a, llm = make(tmp_path, RULES)
    a.chat("u1", "s1", "Quick note on Globex: procurement lead Tom Hale wants invoices in EUR.", at(0))

    b, llm = make(tmp_path, RULES)
    b.chat("u1", "s2", "When did the Globex pilot start?", at(8))
    mem = memory_of(llm)
    assert "Tom Hale" in mem                                         # real facts are there
    assert "start" not in mem.lower().replace("not on record", "")   # but no start-date fact exists
    assert "END OF MEMORY. Anything not listed above is NOT on record." in mem
    assert "NEVER guess or invent" in llm.last_system


def test_assistant_text_is_never_saved_as_fact(tmp_path):
    a, llm = make(tmp_path, RULES)
    a.chat("u1", "s1", "Quick note on Globex: procurement lead Tom Hale wants invoices in EUR.", at(0))
    store = a._store("u1")
    assert all("ok" != d.page_content for d in store.search_all("ok", 50))   # the model's reply ("ok") is not memory
