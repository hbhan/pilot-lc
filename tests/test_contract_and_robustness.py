"""CONTRACT + ROBUSTNESS - rules from the brief that are not one of the six complaints.

Each test names the brief rule it checks.
"""
import re
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from tests.helpers import at, make, memory_of

RULES = {
    "dana ruiz": [{"account": "Acme", "person": "Dana Ruiz", "kind": "contact", "key": "contact.cfo",
                   "text": "Dana Ruiz is the CFO at Acme."}],
    "marcus lee": [{"account": "Acme", "person": "Marcus Lee", "kind": "contact", "key": "contact.cfo",
                    "text": "Marcus Lee is the CFO at Acme."}],
}


def test_end_session_may_be_skipped(tmp_path):
    """'end_session may be skipped': we never call it, restart, and nothing is lost (learning is per message)."""
    a, _ = make(tmp_path, RULES)
    a.chat("u1", "s1", "Dana Ruiz is the CFO at Acme.", at(0))               # tab closed here - no end_session
    b, llm = make(tmp_path, RULES)
    b.chat("u1", "s2", "Who is the CFO at Acme?", at(1))
    assert "Dana Ruiz" in memory_of(llm)


def test_two_users_never_share(tmp_path):
    """'Two different users never share information': separate FAISS folders per user."""
    a, llm = make(tmp_path, RULES)
    a.chat("u1", "s1", "Dana Ruiz is the CFO at Acme.", at(0))
    a.chat("u2", "s1", "Who is the CFO at Acme?", at(1))
    assert "Dana" not in memory_of(llm) and "Acme" not in memory_of(llm)


def test_time_comes_from_ts(tmp_path):
    """'Time comes from ts, never the system clock.'"""
    a, llm = make(tmp_path)
    ts = at(days=1000)                                                        # a date far from "now"
    a.chat("u1", "s1", "hello there, quick question", ts)
    assert ts.strftime("%Y-%m-%d") in llm.last_system                         # prompt shows OUR date
    source = "".join(p.read_text() for p in Path("pilot_lc").glob("*.py"))
    assert not re.search(r"datetime\.(now|utcnow|today)|time\.time\(|date\.today", source)


def test_correction_replaces_old_fact(tmp_path):
    """A corrected fact (new CFO) must replace the old one, not sit next to it."""
    a, llm = make(tmp_path, RULES)
    a.chat("u1", "s1", "Dana Ruiz is the CFO at Acme.", at(0))
    a.chat("u1", "s2", "Marcus Lee is now the CFO at Acme.", at(9))
    a.chat("u1", "s3", "Who is Acme's CFO?", at(10))
    mem = memory_of(llm)
    assert "Marcus Lee" in mem and "Dana Ruiz" not in mem


def test_token_limit_never_exceeded_under_load(tmp_path):
    """'every reply must report input_tokens <= 8000' - 60 turns of large messages, many facts."""
    rules = {"filler": [{"account": f"Client{i}", "kind": "note", "text": f"Client{i} note " + "word " * 50} for i in range(8)]}
    a, _ = make(tmp_path, rules)
    worst = 0
    for i in range(60):
        r = a.chat("u1", "s1", "filler text about Client1 and Client2 and Client3 " + "blah " * 400, at(0, i))
        worst = max(worst, r.input_tokens)
    assert worst <= 8000


def test_model_outage_does_not_crash_or_lose_facts(tmp_path):
    """Provider failure: graceful reply, and the user's facts were already saved (learn-first ordering)."""
    a, _ = make(tmp_path, RULES, llm=RunnableLambda(lambda _: 1 / 0))
    r = a.chat("u1", "s1", "Dana Ruiz is the CFO at Acme.", at(0))
    assert "saved" in r.text
    b, llm = make(tmp_path, RULES)
    b.chat("u1", "s2", "Who is Acme's CFO?", at(1))
    assert "Dana Ruiz" in memory_of(llm)


def test_extractor_failure_keeps_the_words(tmp_path):
    """Bad extractor output must degrade to a raw 'note', never silently drop what the user said."""
    a, _ = make(tmp_path, extractor=RunnableLambda(lambda _: "this is not json"))
    a.chat("u1", "s1", "Call with Acme: their CFO wants a one-page summary.", at(0))
    texts = [d.page_content for d in a._store("u1").search_all("one-page summary", 5)]
    assert any("one-page summary" in t for t in texts)


def test_repeated_fact_is_stored_once(tmp_path):
    """Users repeat themselves. The same sentence must not be stored (and billed in every prompt) twice."""
    a, llm = make(tmp_path, RULES)
    a.chat("u1", "s1", "Dana Ruiz is the CFO at Acme.", at(0))
    a.chat("u1", "s2", "Reminder: Dana Ruiz is the CFO at Acme.", at(1))
    a.chat("u1", "s3", "Who is the CFO at Acme?", at(2))
    assert memory_of(llm).count("Dana Ruiz is the CFO at Acme.") == 1
