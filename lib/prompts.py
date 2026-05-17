"""Prompt templates for the LLM pairwise judge.

The user-side template is anchored on Lihi's playbook
(`thesis/latex/Raz - Ensemble models R2R/SCaLA-26/for_raz.tex`, step 9)
but enriched on 2026-05-14 with:

  1. A system message that anchors the LLM to PERSUADE 2.0's actual
     scope: U.S. students grades 6-12, argumentative writing, 15
     prompts covering social/technology/education/policy topics, both
     independent and source-based tasks. Without this anchor the LLM
     defaults to college-level expectations and penalizes grade-level
     prose unfairly.

  2. Three extra criteria from the PERSUADE rubric tradition that
     Lihi's playbook had omitted -- vocabulary, syntax, conventions.
     The human raters in the corpus do score on those dimensions
     (they appear in the 6 sub-scores: cohesion, syntax, vocabulary,
     phraseology, grammar, conventions), so omitting them from the
     LLM judge biased it toward only high-level argumentation. With
     them included, the LLM more closely mimics how the corpus was
     actually graded.

  3. JSON output (`{"winner": "A"}`) instead of a bare letter, so the
     parser is unambiguous and we can extend the response with extra
     fields (confidence, rationale) later without breaking cache keys.

Everything routes through `lib/llm_judge.py`; the cache key hashes the
full text of both messages, so any edit to either template invalidates
the cache automatically.
"""

PAIRWISE_SYSTEM_MESSAGE = """You are an experienced writing teacher evaluating student argumentative essays from the PERSUADE 2.0 corpus. The essays were written by U.S. secondary school students on prompts about social issues, technology, education, and policy. Some are independent writing tasks; others require the student to write based on provided source material. Judge each essay on the quality of its writing alone."""


PAIRWISE_USER_TEMPLATE = """You are evaluating two student essays.

Criterion:
overall argumentative writing quality, including:
- coherence
- clarity
- persuasiveness
- organization
- use of evidence
- vocabulary and word choice
- syntax and sentence variety
- grammar and conventions

Essay A:
{essay_a}

Essay B:
{essay_b}

Which essay is stronger overall?

Return ONLY a JSON object in this exact form:
{{"winner": "A"}}
or
{{"winner": "B"}}"""


def format_pairwise_messages(essay_a: str, essay_b: str) -> list[dict]:
    """Build the message list sent to the LLM as `messages=[...]`.

    Returns a system + user pair. The returned objects are the same
    shape OpenAI and Anthropic both accept (the Anthropic backend in
    `llm_judge.py` passes the system message via the `system=` kwarg
    rather than inside the messages list, but the source content is
    identical).

    The cache key in `llm_judge.py` hashes the concatenation of
    `system + "\\n\\n" + user`, so changes to either piece invalidate
    the cache.
    """
    user_content = PAIRWISE_USER_TEMPLATE.format(essay_a=essay_a, essay_b=essay_b)
    return [
        {"role": "system", "content": PAIRWISE_SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def format_pairwise_prompt(essay_a: str, essay_b: str) -> str:
    """Single-string form -- system + user concatenated.

    Kept for backward compatibility with callers (and for cache keying)
    that operate on a flat prompt string. The `\\n\\n` separator is
    what the cache key hashes.
    """
    user_content = PAIRWISE_USER_TEMPLATE.format(essay_a=essay_a, essay_b=essay_b)
    return f"{PAIRWISE_SYSTEM_MESSAGE}\n\n{user_content}"
