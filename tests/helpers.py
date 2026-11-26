"""Shared test plumbing: build an Assistant wired to the offline doubles (no key, no network)."""
from datetime import datetime, timedelta

from pilot_lc import Assistant
from pilot_lc.offline import HashingEmbeddings, RecordingChatModel, scripted_extractor

T0 = datetime(2026, 3, 2, 9, 0)          # a fixed "Monday 09:00" used as the simulated clock


def at(days=0, minutes=0):
    """A timestamp `days`/`minutes` after T0. Tests control time explicitly, like the harness."""
    return T0 + timedelta(days=days, minutes=minutes)


def make(data_dir, rules=None, llm=None, extractor=None, config=None):
    """Return (assistant, recording_llm). Calling make() again on the SAME data_dir simulates a restart:
    a brand-new Assistant object that must rebuild everything from disk."""
    llm = llm or RecordingChatModel()
    a = Assistant(str(data_dir), config or {"model": "offline"}, llm=llm,
                  extractor_llm=extractor or scripted_extractor(rules),
                  embeddings=HashingEmbeddings())
    return a, llm


def memory_of(llm) -> str:
    """The MEMORY section of the last prompt the (fake) model received."""
    return llm.last_system.split("MEMORY\n", 1)[1]
