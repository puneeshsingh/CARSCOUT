"""
Input gates for the CarScout Streamlit UI, run in order before the agent:

1. check_moderation() - OpenAI Moderation API. Blocks profanity, harassment,
   violence, and other harmful content. A purpose-built classifier, not a
   keyword list or embedding similarity - moderation needs the former.
2. check_relevance() - three layers, cheapest first:
   a. word-count floor - rejects inputs under MIN_WORD_COUNT words outright
      ("vehicle" is 1 word; no point embedding it).
   b. generic-terms blocklist - rejects inputs made up entirely of
      car-adjacent but contentless words ("car problem"), even if they
      clear the word-count floor.
   c. embedding similarity against vehicle-issue anchor phrases - catches
      everything else that's still vague or off-topic ("hello", "test").

Both check_moderation and check_relevance fail closed: if the underlying API
call errors, the gate blocks rather than silently letting
unmoderated/unfiltered input through.
"""

import logging
import math
import string

from openai import OpenAI

logger = logging.getLogger("carscout_guards")

_client = OpenAI()

MODERATION_MODEL = "omni-moderation-latest"

RELEVANCE_MODEL = "text-embedding-3-small"
RELEVANCE_ANCHORS = [
    "a mechanical, electrical, or safety problem with a vehicle",
    "a symptom or malfunction a car is experiencing",
    "a question about known reliability issues for a car",
]
# Empirically measured with text-embedding-3-small: off-topic inputs like
# "hello" (0.22), "test" (0.29), and unrelated questions (0.06) all scored
# well under 0.35, while real vehicle symptoms scored 0.44-0.52. 0.35 sits
# in the gap with margin on both sides.
RELEVANCE_THRESHOLD = 0.35

# Below this many words, an input can't describe an actual symptom - reject
# before even running the (more expensive) embedding check. Set to 2 (not 3)
# because real two-word symptoms ("brake noise", "engine stalling") must
# still pass; two-word inputs that are still too generic ("car problem") are
# instead caught by GENERIC_TERMS_BLOCKLIST below.
MIN_WORD_COUNT = 2

# Car-adjacent vocabulary that describes nothing specific on its own. Blocks
# inputs made up entirely of these words (e.g. "car problem"), even if they
# clear MIN_WORD_COUNT or would otherwise score above RELEVANCE_THRESHOLD.
GENERIC_TERMS_BLOCKLIST = {
    "vehicle",
    "car",
    "issue",
    "problem",
    "help",
    "check",
    "info",
    "question",
}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def _normalized_words(text: str) -> list[str]:
    return [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]


def check_moderation(text: str) -> dict:
    """Returns {"status": "ok" | "flagged" | "error", "categories": [...], "error"?: str}.

    "flagged" and "error" both mean: block the run. Categories are for
    server-side logging only - never surface them to the end user.
    """
    try:
        response = _client.moderations.create(model=MODERATION_MODEL, input=text)
        result = response.results[0]
        categories = [name for name, value in result.categories.model_dump().items() if value]
        status = "flagged" if result.flagged else "ok"
        logger.info("MODERATION -> status=%s categories=%s", status, categories)
        return {"status": status, "categories": categories}
    except Exception as e:
        logger.error("MODERATION check failed, failing closed: %s", e)
        return {"status": "error", "categories": [], "error": str(e)}


def check_relevance(text: str) -> dict:
    """Returns {"status": "relevant" | "irrelevant" | "error", "score": float | None, "error"?: str}.

    "score" is None when blocked by the word-count or blocklist layer -
    those never reach the embedding call.
    """
    words = _normalized_words(text)

    if len(words) < MIN_WORD_COUNT:
        logger.info("RELEVANCE -> status=irrelevant reason=too_short word_count=%d text=%r", len(words), text)
        return {"status": "irrelevant", "score": None}

    if all(word in GENERIC_TERMS_BLOCKLIST for word in words):
        logger.info("RELEVANCE -> status=irrelevant reason=generic_only text=%r", text)
        return {"status": "irrelevant", "score": None}

    try:
        response = _client.embeddings.create(model=RELEVANCE_MODEL, input=[text, *RELEVANCE_ANCHORS])
        vectors = [d.embedding for d in response.data]
        query_vector, anchor_vectors = vectors[0], vectors[1:]
        best_score = max(_cosine_similarity(query_vector, anchor) for anchor in anchor_vectors)
        status = "relevant" if best_score >= RELEVANCE_THRESHOLD else "irrelevant"
        logger.info("RELEVANCE -> status=%s score=%.4f text=%r", status, best_score, text)
        return {"status": status, "score": best_score}
    except Exception as e:
        logger.error("RELEVANCE check failed, failing closed: %s", e)
        return {"status": "error", "score": None, "error": str(e)}
