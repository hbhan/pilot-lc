"""PROBLEM 5 - "I asked about Contoso and it mentioned a contact who belongs to a different client."

ROOT CAUSE   Facts of all clients sit in ONE flat context, so the model sees two 'Priya's and picks one.
HOW WE FIX   fact_store.for_account filters FAISS by metadata {"account": X}: other clients' facts are not even
             candidates when the user names a client (HARD scope). When no client is named (SOFT scope) facts are
             shown with an explicit [Client] label and the prompt rule 2 forbids mixing.
WHAT THE TEST PROVES  Naming Contoso => Fabrikam's Priya is absent from the prompt. Not naming a client =>
             both appear, but clearly labelled by client.
"""
from tests.helpers import at, make, memory_of

RULES = {
    "priya nair": [{"account": "Contoso", "person": "Priya Nair", "kind": "contact",
                    "text": "Priya Nair is the CFO at Contoso and cares most about payback period."}],
    "priya menon": [{"account": "Fabrikam", "person": "Priya Menon", "kind": "contact",
                     "text": "Priya Menon is the procurement lead at Fabrikam and dislikes long contracts."}],
}


def test_named_client_is_a_hard_boundary(tmp_path):
    a, llm = make(tmp_path, RULES)
    a.chat("u1", "s1", "Met with Contoso. Their CFO Priya Nair cares about payback period.", at(0))
    a.chat("u1", "s1", "Met with Fabrikam. Their procurement lead Priya Menon hates long contracts.", at(0, 5))

    a.chat("u1", "s2", "Prep me for my Contoso call. What does Priya care about?", at(1))
    mem = memory_of(llm)
    assert "Priya Nair" in mem and "payback" in mem
    assert "Menon" not in mem and "long contracts" not in mem     # the other client's Priya cannot leak


def test_unnamed_client_results_are_labelled(tmp_path):
    a, llm = make(tmp_path, RULES)
    a.chat("u1", "s1", "Met with Contoso. Their CFO Priya Nair cares about payback period.", at(0))
    a.chat("u1", "s1", "Met with Fabrikam. Their procurement lead Priya Menon hates long contracts.", at(0, 5))
    a.chat("u1", "s2", "What does Priya care about?", at(1))         # ambiguous on purpose
    mem = memory_of(llm)
    assert "[Contoso]" in mem and "[Fabrikam]" in mem                # both shown, each labelled
    assert "ask instead of assuming" in llm.last_system              # prompt rule 5 tells the model to ask
