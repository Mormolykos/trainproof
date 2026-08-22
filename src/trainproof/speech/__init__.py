"""Speech/TTS domain pack - dataset and tokenizer preflight.

The trainproof core (epoch log linter, rules, reports) is model-agnostic;
domain-specific checks live in packs like this one so future packs
(e.g. LLM fine-tuning) plug in beside it.

`check_data` is resolved lazily. It needs numpy, soundfile and ttsproof, and
soundfile needs libsndfile, a C library. Importing it eagerly here made the
whole CLI - including `trainproof epoch`, which reads text logs and needs none
of that - refuse to start without an audio stack installed. Those three are now
the `trainproof[speech]` extra, and nothing pulls them in until a speech check
is actually called.

`check_tokenizer` has no such dependencies and is imported normally.
"""

from typing import Any

from .tokenizer import check_tokenizer

__all__ = ["check_data", "check_tokenizer"]


def __getattr__(name: str) -> Any:
    if name == "check_data":
        from .data import check_data
        return check_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
