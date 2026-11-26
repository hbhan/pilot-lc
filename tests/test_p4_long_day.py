"""PROBLEM 4 - "After a day of back-to-back calls it starts ignoring what I told it that morning."

ROOT CAUSE   Inside ONE session the verbatim history keeps growing until the earliest messages are evicted
             (or drowned out).
HOW WE FIX   (a) trim_messages in the chain keeps the verbatim window to `history_tokens` - bounded, newest first.
             (b) Everything said is ALSO saved as facts on every message, so the evicted morning call is still
                 retrievable from FAISS.
             (c) With no client named in the question, retriever.build_memory does a labelled similarity
                 search across all clients (soft scope).
WHAT THE TEST PROVES  After ~42 long messages the first call is NO LONGER in the verbatim history (trimmed)
             but IS in the memory block, and the prompt is still within budget.
"""
from tests.helpers import at, make, memory_of

CALLS = ["Initech", "Umbrella", "Hooli", "Stark", "Wayne", "Tyrell", "Wonka"]
RULES = {f"finished the call with {c.lower()}": [{"account": c, "kind": "event",
         "text": f"{c}'s data migration deadline is {'2026-04-14' if c == 'Initech' else '2026-06-01'}."}] for c in CALLS}


def test_morning_call_survives_a_long_day(tmp_path):
    a, llm = make(tmp_path, RULES)
    a.start_session("u1", "day", at(0))
    n = 0
    for c in CALLS:
        a.chat("u1", "day", f"Just finished the call with {c}. " + "They discussed rollout and risks. " * 8, at(0, 30 * n)); n += 1
        for k in range(5):
            a.chat("u1", "day", f"More on {c}: " + "follow-up planning and stakeholder concerns. " * 7, at(0, 30 * n)); n += 1

    reply = a.chat("u1", "day", "What was the data migration deadline from my very first call this morning?", at(1))

    history_text = " ".join(m.content for m in llm.last_history)
    assert "Initech" not in history_text                     # (a) the morning call was trimmed out of the window
    assert "2026-04-14" in memory_of(llm)                    # (b)+(c) but memory still has it
    assert reply.input_tokens <= 8000                        # and we stayed in budget
