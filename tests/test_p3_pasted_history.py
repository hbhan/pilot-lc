"""PROBLEM 3 - "Somebody pasted all their old chats in. Worked a week, then slow, expensive, mixing up details."

ROOT CAUSE   Stuffing history into every call: cost and latency grow with history; the 8k cap forces blind
             truncation; facts get buried mid-context and mixed across clients.
HOW WE FIX   Pasted text is split (RecursiveCharacterTextSplitter) and each chunk is EXTRACTED into FAISS facts
             once (assistant._learn). The reply call sees only a clipped excerpt (assistant._clip_middle).
             Later questions retrieve just the relevant fact. Cost per reply stays flat.
WHAT THE TEST PROVES  A ~40k-character paste (a) is ingested in several chunks, (b) keeps the paste turn under
             the 8,000-token limit, (c) a needle fact from the MIDDLE of the paste is retrievable next turn,
             (d) the next turn's prompt is small again.
"""
from langchain_core.runnables import RunnableLambda

from pilot_lc.offline import scripted_extractor
from tests.helpers import at, make, memory_of

RULES = {"soylent's renewal date": [{"account": "Soylent", "kind": "status", "key": "status.renewal",
                                     "text": "Soylent's renewal date is 2026-09-30."}]}


def test_big_paste_is_ingested_chunked_and_bounded(tmp_path):
    inner, calls = scripted_extractor(RULES), []
    counting = RunnableLambda(lambda pv: (calls.append(1), inner.invoke(pv))[1])   # count extraction calls

    a, llm = make(tmp_path, extractor=counting)
    lines = [f"[2026-02-{1 + i % 27:02d}] Standup {i}: sync with Initech about pricing review and rollout." for i in range(450)]
    lines.insert(225, "[2026-02-14] Soylent's renewal date is 2026-09-30, confirmed by procurement.")
    paste = "Here are all my old notes:\n" + "\n".join(lines)
    assert len(paste) > 35_000

    reply = a.chat("u1", "s1", paste, at(0))
    assert len(calls) >= 4                                   # (a) split into several small extraction calls
    assert reply.input_tokens <= 8000                        # (b) within the contract limit
    assert "long message shortened" in llm.calls[-1][-1].content   # the reply model only saw an excerpt

    reply2 = a.chat("u1", "s1", "When is Soylent's renewal?", at(0, 2))
    assert "2026-09-30" in memory_of(llm)                    # (c) needle recovered from FAISS, not from the paste
    assert reply2.input_tokens < reply.input_tokens          # (d) cost back down - it does not grow with history
