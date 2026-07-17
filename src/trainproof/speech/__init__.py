"""Speech/TTS domain pack — dataset and tokenizer preflight.

The trainproof core (epoch log linter, rules, reports) is model-agnostic;
domain-specific checks live in packs like this one so future packs
(e.g. LLM fine-tuning) plug in beside it.
"""

from .data import check_data
from .tokenizer import check_tokenizer
