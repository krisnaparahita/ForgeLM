"""Prompt token budget shared by the safety gate and the LLM judge.

Both gates used to tokenize the prompt under test with a hard-coded
``truncation=True, max_length=1024``.  Tokenizers truncate on the right, so an
over-long prompt silently lost its tail (usually the actual request) and the
row was still scored.  The helpers here derive the budget from the model's
context window and make an over-long prompt an explicit error instead.
"""

import logging
from typing import Any

logger = logging.getLogger("forgelm.token_budget")

# Used only when neither the model config nor the tokenizer reports a usable
# context length.  Matches the literal the gates used before.
FALLBACK_PROMPT_TOKEN_BUDGET = 1024

# HF tokenizers report ``int(1e30)`` for "no limit"; treat anything above this as unset.
_UNSET_LENGTH_THRESHOLD = 1_000_000

_fallback_warned = False


class PromptTooLongError(ValueError):
    """The prompt does not fit in the model's context window next to ``max_new_tokens``."""


def prompt_token_budget(model: Any, tokenizer: Any, max_new_tokens: int) -> int:
    """Return how many prompt tokens fit: context length minus ``max_new_tokens``."""
    global _fallback_warned
    context = None
    for owner, attr in (
        (getattr(model, "config", None), "max_position_embeddings"),
        (tokenizer, "model_max_length"),
    ):
        value = getattr(owner, attr, None)
        if isinstance(value, int) and 0 < value < _UNSET_LENGTH_THRESHOLD:
            context = value
            break
    if context is None:
        if not _fallback_warned:
            logger.warning(
                "Model exposes no context length; using a %d-token prompt budget.", FALLBACK_PROMPT_TOKEN_BUDGET
            )
            _fallback_warned = True
        return FALLBACK_PROMPT_TOKEN_BUDGET
    return max(context - max_new_tokens, 1)


def check_prompt_fits(input_ids: Any, budget: int) -> None:
    """Raise :class:`PromptTooLongError` when the tokenized prompt exceeds ``budget``."""
    length = input_ids.shape[1]
    if length > budget:
        raise PromptTooLongError(f"prompt is {length} tokens, over the {budget}-token budget; not truncated")
