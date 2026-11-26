"""Walkthrough: replays the SIX user complaints and prints what the model actually received.

    python scripts/walkthrough.py            # offline: no key, no network (fake LLM + fake embeddings)
    python scripts/walkthrough.py --live     # real OpenAI (needs OPENAI_API_KEY); prints real replies too
    python scripts/walkthrough.py --only 5   # run one problem

For each problem you will see: (1) what the user said, (2) the MEMORY block the model was given,
(3) the reply, (4) input tokens vs the 8,000 limit.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # allow running from any folder
warnings.filterwarnings("ignore")

from pilot_lc import Assistant                                   # noqa: E402
from pilot_lc.offline import HashingEmbeddings, RecordingChatModel, scripted_extractor  # noqa: E402

MON = datetime(2026, 3, 2, 9, 0)                                 # simulated clock; we pass it as `ts`


def t(days=0, minutes=0) -> datetime:
    return MON + timedelta(days=days, minutes=minutes)


# What a GOOD extraction LLM would return for each scripted message (used only in offline mode).
OFFLINE_RULES = {
    "one-page": [{"account": "Acme", "person": "Dana Ruiz", "kind": "preference",
                  "text": "Acme's CFO Dana Ruiz wants every proposal as a one-page summary."}],
    "dana ruiz is the cfo": [{"account": "Acme", "person": "Dana Ruiz", "kind": "contact", "key": "contact.cfo",
                              "text": "Dana Ruiz is the CFO at Acme."}],
    "tom hale": [{"account": "Globex", "person": "Tom Hale", "kind": "contact",
                  "text": "Tom Hale is the procurement lead at Globex and wants invoices in EUR."}],
    "soylent's renewal date": [{"account": "Soylent", "kind": "status",
                                "text": "Soylent's renewal date is 2026-09-30."}],
    "finished the call with initech": [{"account": "Initech", "kind": "event",
                                 "text": "Initech's data migration deadline is 2026-04-14."}],
    "priya nair": [{"account": "Contoso", "person": "Priya Nair", "kind": "contact",
                    "text": "Priya Nair is the CFO at Contoso and cares most about payback period."}],
    "priya menon": [{"account": "Fabrikam", "person": "Priya Menon", "kind": "contact",
                     "text": "Priya Menon is the procurement lead at Fabrikam and dislikes long contracts."}],
}


def build(live: bool, folder: str):
    if live:
        return Assistant(folder, {"model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"}), None
    llm = RecordingChatModel()
    return Assistant(folder, {"model": "offline"}, llm=llm, embeddings=HashingEmbeddings(),
                     extractor_llm=scripted_extractor(OFFLINE_RULES)), llm


def say(a, user, session, text, ts, show=True):
    """Send one message and (optionally) print what the model saw."""
    shown = text if len(text) < 150 else text[:100] + f" ...[{len(text):,} chars]..."
    print(f"  USER [{ts:%a %H:%M}]: {shown}")
    r = a.chat(user, session, text, ts)
    if show:
        print("  --- MEMORY BLOCK GIVEN TO THE MODEL " + "-" * 38)
        print("  " + a.last_memory.replace("\n", "\n  "))
        print("  " + "-" * 74)
        print(f"  REPLY: {r.text[:300]}")
        print(f"  INPUT TOKENS: {r.input_tokens:,}  (limit 8,000)")
    return r


def header(n, title, complaint):
    print("\n" + "=" * 78 + f"\nPROBLEM {n}: {title}\n  Complaint: {complaint}\n" + "=" * 78)


def p1(a):
    header(1, "Forgotten preference", "Told it Monday the CFO wants one page; Thursday it wrote five")
    say(a, "p1", "mon", "Just off a call with Acme. Their CFO, Dana Ruiz, wants every proposal as a one-page summary.", t(0), show=False)
    for i in range(3):   # unrelated sessions in between
        say(a, "p1", f"filler{i}", f"Call with Initech about pricing, session {i}. Long discussion of details.", t(1, i), show=False)
    print("  ...later, brand-new session on Thursday:")
    say(a, "p1", "thu", "Outline the renewal proposal for Acme's CFO.", t(3))


def p2(a):
    header(2, "Re-explaining who is who", "Every morning I have to re-explain who is who")
    say(a, "p2", "mon", "Dana Ruiz is the CFO at Acme.", t(0), show=False)
    say(a, "p2", "tue", "Tom Hale is our procurement lead at Globex, he wants invoices in EUR.", t(1), show=False)
    print("  ...next morning, new session:")
    say(a, "p2", "wed", "Remind me who Tom Hale is and who the CFO at Acme is.", t(2))


def p3(a):
    header(3, "Pasting all old chats", "Pasted everything; worked a week, then slow, costly, confused")
    lines = [f"[2026-02-{1 + i % 27:02d}] Standup {i}: sync with Initech about pricing review and rollout." for i in range(450)]
    lines.insert(225, "[2026-02-14] Soylent's renewal date is 2026-09-30, confirmed by procurement.")
    r1 = say(a, "p3", "s1", "Here are all my old notes:\n" + "\n".join(lines), t(0), show=False)
    print(f"  PASTE TURN input tokens: {r1.input_tokens:,} (paste was ~{len(lines) * 90:,} chars, ingested in chunks)")
    say(a, "p3", "s1", "When is Soylent's renewal?", t(0, 2))


def p4(a):
    header(4, "Ignores the morning after a long day", "After back-to-back calls it ignores what I told it this morning")
    calls = ["Initech", "Umbrella", "Hooli", "Stark", "Wayne", "Tyrell", "Wonka"]
    n = 0
    for c in calls:
        first = (f"Just finished the call with {c}. " + ("Their data migration deadline is 2026-04-14. " if c == "Initech" else "") + "We covered rollout and risks. " * 8)
        say(a, "p4", "day", first, t(0, 30 * n), show=False); n += 1
        for _ in range(3):
            say(a, "p4", "day", f"More on {c}: follow-up planning and stakeholder concerns. " * 6, t(0, 30 * n), show=False); n += 1
    hist = a._history("p4::day").messages
    print(f"  Session now holds {len(hist)} messages; only the newest ~{a.cfg.history_tokens} tokens go verbatim into the prompt.")
    say(a, "p4", "day", "What was the data migration deadline from my very first call this morning?", t(1))


def p5(a):
    header(5, "Contact from the wrong client", "Asked about Contoso, it mentioned someone from a different client")
    say(a, "p5", "s1", "Met with Contoso. Their CFO Priya Nair cares about payback period.", t(0), show=False)
    say(a, "p5", "s1", "Met with Fabrikam. Their procurement lead Priya Menon hates long contracts.", t(0, 5), show=False)
    print("  (a) client NAMED -> hard scope: Fabrikam's Priya must not appear")
    say(a, "p5", "s2", "Prep me for my Contoso call. What does Priya care about?", t(1))
    print("  (b) client NOT named -> labelled results, model is told to ask which one")
    say(a, "p5", "s2", "What does Priya care about?", t(1, 5))


def p6(a):
    header(6, "Invented fact", "Asked when the Globex pilot started; it gave a confident date I never mentioned")
    say(a, "p6", "s1", "Quick note on Globex: procurement lead Tom Hale wants invoices in EUR.", t(0), show=False)
    print("  The user never said when the pilot started. Look for: no date fact + the 'NOT on record' boundary.")
    say(a, "p6", "s2", "When did the Globex pilot start?", t(8))
    print("  EXPECTED live behaviour: the model says it has no record of the start date and asks for it.")


PROBLEMS = {1: p1, 2: p2, 3: p3, 4: p4, 5: p5, 6: p6}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use real OpenAI instead of offline doubles")
    ap.add_argument("--only", type=int, choices=PROBLEMS, help="run a single problem number")
    args = ap.parse_args()
    if not args.live:
        print("[OFFLINE MODE] fake LLM/embeddings: replies are stubs; focus on the MEMORY block and token counts.")
    with tempfile.TemporaryDirectory() as folder:
        assistant, _ = build(args.live, folder)
        for n, fn in PROBLEMS.items():
            if args.only in (None, n):
                fn(assistant)
