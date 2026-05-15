"""External benchmark sources beyond Chatbot Arena and Open LLM Leaderboard.

Each module here fetches an independent leaderboard / index, normalizes it to
the same 0-100 scale, and returns a ``dict[str, float]`` keyed by HuggingFace
model id (or a list of synonyms).

The functions are intentionally defensive: if a source is unreachable or
returns malformed data, they log a warning and return an empty dict so the
main benchmark merge pipeline does not abort.
"""

from whichllm.models.benchmark_sources.aa_index import fetch_aa_index_scores
from whichllm.models.benchmark_sources.aider import fetch_aider_polyglot_scores
from whichllm.models.benchmark_sources.livebench import fetch_livebench_scores
from whichllm.models.benchmark_sources.vision import fetch_vision_scores

# Newest curated-fallback date across all sources. Live scrapes are merged
# on top when reachable, but they frequently are not (the leaderboard
# spaces change their JSON shape), so the user-visible ranking is anchored
# to this snapshot. Surface it in the CLI so a stale recommendation is
# self-evident rather than silently trusted. Bump this whenever any
# *_FALLBACK_* dict is refreshed.
BENCHMARK_SNAPSHOT = "2026-05"

__all__ = [
    "BENCHMARK_SNAPSHOT",
    "fetch_aa_index_scores",
    "fetch_aider_polyglot_scores",
    "fetch_livebench_scores",
    "fetch_vision_scores",
]
