"""COL 764 Assignment 4 - Iterative Retrieval for Multi-hop QA.

Your agent. The harness calls ``run(question, tools)`` once per question and expects

    {"answer": "<short answer>", "citations": ["<doc_id>", ...]}   # at most 3 doc_ids are scored

The tools (handout Section 2.2):

    tools.search(query, k=5)        -> list[Passage]   BM25, 1 <= k <= 10      (max 5 calls / question)
    tools.llm(prompt, max_tokens)   -> str             temperature 0           (max 6 calls / question)
    tools.budget_left()             -> dict            {"search_calls", "llm_calls", "seconds"}

    Passage has .doc_id, .title, .text (and str(p) gives "[doc_id] title: text").
    Over-budget calls raise BudgetExhausted; prompts over 3,000 tokens raise PromptTooLong (not counted).

AS GIVEN, THIS FILE IS THE SINGLE-SHOT BASELINE B1: ``should_stop`` always returns True, so the loop
searches once and answers. Your job (handout Section 3):

    (a) implement next_query()   - what to search for next
    (b) implement should_stop()  - when the evidence is enough (and when the budget runs out)
    (c) improve answer()         - short answer + the right citations

You may change anything in this file (and add helper modules next to it), as long as ``run`` keeps its
signature and the loop stays genuinely iterative: later queries must depend on what was retrieved earlier.
Only import the Python standard-library modules listed in README.md, numpy, and ``a4``.
"""

from a4 import BudgetExhausted, PromptTooLong, Passage


# ======================================================================================================
# Notes: everything the agent has gathered so far for one question
# ======================================================================================================
class Notes:
    """Passages retrieved so far (de-duplicated, in the order first seen), the queries issued, and any
    short reasoning steps you want to keep (e.g. "The bridge entity is X")."""

    def __init__(self):
        self.passages: dict[str, Passage] = {}
        self.queries: list[str] = []
        self.thoughts: list[str] = []

    def add(self, passages: list[Passage]) -> None:
        for p in passages:
            if p.doc_id not in self.passages:
                self.passages[p.doc_id] = p

    def top(self, n: int) -> list[Passage]:
        return list(self.passages.values())[:n]

    def as_text(self, max_passages: int = 8) -> str:
        return "\n".join(str(p) for p in self.top(max_passages))


# ======================================================================================================
# (a) next_query - TODO
# ======================================================================================================
def next_query(question: str, notes: Notes, tools) -> str:
    """Return the next search query, based on what has been found so far.

    A typical first version for 2-hop questions:
      1. Ask the LLM which intermediate ("bridge") fact or entity the passages provide and what is still
         missing to answer the question, e.g. "Answer with only the name of the entity we need to look up next."
      2. Store the result in notes.thoughts.
      3. Return a query built from it (e.g. the bridge entity plus the remaining part of the question).

    Never just return `question` again: a repeated query returns the same passages.
    """
    raise NotImplementedError("TODO: implement next_query (handout Section 3a)")


# ======================================================================================================
# (b) should_stop - TODO
# ======================================================================================================
def should_stop(question: str, notes: Notes, tools) -> bool:
    """Return True when there is enough evidence to answer.

    Ideas: ask the LLM a yes/no question ("Do these passages contain the answer?"), stop after a fixed
    number of hops, and always stop when tools.budget_left() shows no search (or LLM) calls left.

    As given, it always returns True, which makes this agent the single-shot baseline B1.
    """
    return True  # TODO: replace (handout Section 3b)


# ======================================================================================================
# (c) answer - working baseline, improve it
# ======================================================================================================
ANSWER_PROMPT = """Answer the question using only the passages below.
Reply with the short answer only (a name, place, date or number), not a sentence.

Passages:
{passages}

Question: {question}
Answer:"""


def clean_answer(text: str) -> str:
    """Keep the first line, drop an 'Answer:' prefix and trailing punctuation."""
    line = text.strip().split("\n")[0].strip()
    if line.lower().startswith("answer:"):
        line = line[len("answer:"):].strip()
    return line.strip(" .\"'")


def answer(question: str, notes: Notes, tools) -> dict:
    """Produce {"answer", "citations"}. Baseline: ask the LLM over the top passages; cite the top 2.

    Ideas: cite the passages that actually support each hop (e.g. the one naming the bridge entity and the
    one containing the answer), not just the top-ranked ones.
    """
    for n_passages in (8, 4, 2):  # retry with fewer passages if the prompt is too long
        prompt = ANSWER_PROMPT.format(passages=notes.as_text(n_passages), question=question)
        try:
            out = tools.llm(prompt, max_tokens=32)
            break
        except PromptTooLong:
            continue
    else:
        out = ""
    citations = [p.doc_id for p in notes.top(2)]
    return {"answer": clean_answer(out), "citations": citations}


# ======================================================================================================
# Entry point (called by the harness)
# ======================================================================================================
def run(question: str, tools) -> dict:
    notes = Notes()
    query = question
    try:
        while True:
            notes.queries.append(query)
            notes.add(tools.search(query, k=5))
            if should_stop(question, notes, tools):
                break
            query = next_query(question, notes, tools)
    except BudgetExhausted:
        pass  # out of searches or LLM calls: answer with what we have
    try:
        return answer(question, notes, tools)
    except BudgetExhausted:
        # No LLM calls left: fall back to the title of the best passage (rarely right, but scores > 0).
        top = notes.top(2)
        return {"answer": top[0].title if top else "", "citations": [p.doc_id for p in top]}
