"""Provider-agnostic LLM pairwise judge with on-disk response caching.

The experiment is about *some* LLM judge, not a specific provider. The
`PairwiseJudge` ABC defines the contract; concrete backends wrap the
OpenAI and Anthropic SDKs. New providers slot in by subclassing.

The cache layer wraps every backend uniformly. Cache key includes the
provider, model_id, temperature, and the SHA-256 of the system+user
message text -- any of those changing forces a fresh API call. Cache
values store the raw response alongside the parsed winner, so a
post-hoc audit can recover why the judge said what it said.

Phase 1 makes one call per pair (45 calls per top-10 / 10-window).
Caching makes the second run free. Even at temperature 0, no LLM is
bit-exactly deterministic across calls; the cache is what makes a run
reproducible by replay.

2026-05-14: the prompt switched to system + user messages with JSON
output (`{"winner": "A"}`). The parser tolerates whitespace and
preambles around the JSON object, and falls back to first-letter
heuristics if the model ignores the JSON instruction entirely.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import diskcache

Winner = Literal["A", "B"]


@dataclass(frozen=True)
class JudgeResponse:
    """One LLM judgment, post-parse.

    `winner` is the normalized A/B answer; `raw` is the verbatim text
    the LLM returned (useful for auditing parse failures or running
    qualitative review after the fact).
    """

    winner: Winner
    raw: str
    provider: str
    model_id: str
    temperature: float
    prompt_hash: str
    cached: bool
    elapsed_seconds: float


def _hash_messages(system: str, user: str) -> str:
    """Stable hash of the (system, user) pair used in cache keys."""
    return hashlib.sha256(f"{system}\n\n{user}".encode("utf-8")).hexdigest()


def _cache_key(
    provider: str,
    model_id: str,
    temperature: float,
    system: str,
    user: str,
    reasoning_effort: Optional[str] = None,
) -> str:
    payload = f"{provider}|{model_id}|{temperature}|{_hash_messages(system, user)}"
    if reasoning_effort:
        # Only appended when explicitly non-empty -- preserves backward
        # compat with cache entries written before this parameter
        # existed (those keys have no suffix; treat them as the implicit
        # default that the caller used at the time).
        payload += f"|effort={reasoning_effort}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_WINNER_RE = re.compile(r'"\s*winner\s*"\s*:\s*"\s*([AaBb])\s*"')


def _parse_winner(raw: str) -> Winner:
    """Extract "A" or "B" from a JSON-style response.

    Tolerates:
      - Pure JSON: `{"winner": "A"}`
      - JSON with whitespace / newlines
      - JSON wrapped in markdown fences: ```json {"winner": "A"} ```
      - Preambles before the JSON object
      - All-caps or lower-case `a` / `b`

    Falls back to first-alphabetic-character logic if no JSON-shaped
    `winner` field is found at all -- this preserves compatibility with
    a model that ignored the JSON instruction and just said "A".
    """
    if not raw:
        raise ValueError("Empty response from LLM")

    # 1. Try strict JSON parse first.
    stripped = raw.strip()
    # Strip optional markdown code fences.
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "winner" in obj:
            w = str(obj["winner"]).strip().upper()
            if w in ("A", "B"):
                return w  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    # 2. Regex search for the "winner" field inside a larger blob.
    m = _WINNER_RE.search(raw)
    if m:
        return m.group(1).upper()  # type: ignore[return-value]

    # 3. Fall back: first alphabetic character must be A or B.
    for ch in raw.strip():
        if ch.isalpha():
            up = ch.upper()
            if up in ("A", "B"):
                return up  # type: ignore[return-value]
            break

    raise ValueError(f"Could not parse winner from response: {raw!r}")


class PairwiseJudge(ABC):
    """Abstract base for an LLM-backed pairwise judge.

    Subclasses implement `_call_api(system, user)` returning the raw
    string response. The base class handles caching, parsing, and
    timing.
    """

    provider: str

    def __init__(
        self,
        model_id: str,
        temperature: float,
        cache_dir: Optional[Path] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.temperature = float(temperature)
        # `reasoning_effort` only matters for reasoning-family models
        # (gpt-5, o-series). For legacy models the API rejects the
        # parameter; for reasoning models the OpenAI default is
        # 'medium'. We propagate whatever the caller asked for and let
        # `_call_api` decide whether to forward it.
        self.reasoning_effort = reasoning_effort
        self._cache = (
            diskcache.Cache(str(cache_dir)) if cache_dir is not None else None
        )

    @abstractmethod
    def _call_api(self, system: str, user: str) -> str:
        """Provider-specific API call. Returns the raw response text."""

    def judge(self, system: str, user: str) -> JudgeResponse:
        """Run one pairwise judgment, hitting cache when available."""
        # `effort_for_cache` is the namespace component of the cache
        # key. Treat 'minimal' as the implicit default for reasoning
        # models (the value that was used before this knob existed) so
        # those entries still hit on rerun. Any other explicit effort
        # gets its own namespace.
        effort_for_cache: Optional[str] = None
        if self.reasoning_effort and self.reasoning_effort != "minimal":
            effort_for_cache = self.reasoning_effort
        cache_key = _cache_key(
            self.provider, self.model_id, self.temperature, system, user,
            reasoning_effort=effort_for_cache,
        )
        prompt_hash = _hash_messages(system, user)

        if self._cache is not None and cache_key in self._cache:
            payload = self._cache[cache_key]
            return JudgeResponse(
                winner=payload["winner"],
                raw=payload["raw"],
                provider=self.provider,
                model_id=self.model_id,
                temperature=self.temperature,
                prompt_hash=prompt_hash,
                cached=True,
                elapsed_seconds=0.0,
            )

        t0 = time.monotonic()
        raw = self._call_api(system, user)
        elapsed = time.monotonic() - t0
        winner = _parse_winner(raw)

        if self._cache is not None:
            self._cache[cache_key] = {
                "winner": winner,
                "raw": raw,
                "provider": self.provider,
                "model_id": self.model_id,
                "temperature": self.temperature,
                "prompt_hash": prompt_hash,
                "timestamp_utc": time.time(),
            }

        return JudgeResponse(
            winner=winner,
            raw=raw,
            provider=self.provider,
            model_id=self.model_id,
            temperature=self.temperature,
            prompt_hash=prompt_hash,
            cached=False,
            elapsed_seconds=elapsed,
        )


class OpenAIPairwiseJudge(PairwiseJudge):
    """OpenAI backend. Reads `OPENAI_API_KEY` from the environment."""

    provider = "openai"

    def __init__(
        self,
        model_id: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        cache_dir: Optional[Path] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            cache_dir=cache_dir,
            reasoning_effort=reasoning_effort,
        )
        if "OPENAI_API_KEY" not in os.environ:
            raise EnvironmentError(
                "OPENAI_API_KEY env var is not set; cannot use the OpenAI backend."
            )
        try:
            from openai import OpenAI  # noqa: WPS433 (deferred import is intentional)
        except ImportError as exc:
            raise ImportError(
                "The `openai` package is required for the OpenAI backend. "
                "Install with: pip install openai"
            ) from exc
        self._client = OpenAI()

    def _call_api(self, system: str, user: str) -> str:
        # OpenAI deprecated `max_tokens` in favor of
        # `max_completion_tokens`; newer models (gpt-5+, o-series)
        # reject the legacy name. The newer reasoning-capable models
        # also reject `temperature != 1` -- they only run at default
        # temperature -- so we skip the parameter for them. Cache key
        # still encodes the requested temperature so the audit trail is
        # honest, but no call is made with a value the model rejects.
        is_reasoning_family = (
            self.model_id.startswith("gpt-5")
            or self.model_id.startswith("o1")
            or self.model_id.startswith("o3")
            or self.model_id.startswith("o4")
        )
        kwargs = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Force JSON mode when the model supports it. gpt-4o and
            # later honor `response_format={"type": "json_object"}`;
            # older models silently ignore it.
            "response_format": {"type": "json_object"},
        }
        if is_reasoning_family:
            # gpt-5 and o-series spend internal "thinking tokens" before
            # emitting visible output. With effort=medium they can burn
            # the full token budget on hidden reasoning and return an
            # empty completion, so the ceiling has to scale up with the
            # effort level.
            effort = self.reasoning_effort or "minimal"
            kwargs["reasoning_effort"] = effort
            # Token budget tiered to leave room for both reasoning and
            # the ~30-token JSON answer at the end.
            effort_to_budget = {
                "minimal": 2048,
                "low": 4096,
                "medium": 8192,
                "high": 16384,
            }
            kwargs["max_completion_tokens"] = effort_to_budget.get(effort, 4096)
        else:
            # Legacy chat models: 32 tokens is plenty for `{"winner":"A"}`.
            kwargs["max_completion_tokens"] = 32
            kwargs["temperature"] = self.temperature
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class AnthropicPairwiseJudge(PairwiseJudge):
    """Anthropic backend. Reads `ANTHROPIC_API_KEY` from the environment."""

    provider = "anthropic"

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        cache_dir: Optional[Path] = None,
        max_tokens: int = 32,
    ) -> None:
        super().__init__(model_id=model_id, temperature=temperature, cache_dir=cache_dir)
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY env var is not set; cannot use the Anthropic backend."
            )
        try:
            import anthropic  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "The `anthropic` package is required for the Anthropic backend. "
                "Install with: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic()
        self._max_tokens = max_tokens

    def _call_api(self, system: str, user: str) -> str:
        # Anthropic's newer flagship models (e.g. Claude Opus 4.7)
        # deprecated the `temperature` parameter: the API returns 400 if
        # we send it. Older Claude versions (Sonnet 4.6, Haiku 4.5) still
        # accept it normally. Cache key still encodes the requested
        # temperature so the audit trail stays honest, but the wire call
        # omits the parameter for models that reject it.
        no_temperature = self.model_id.startswith("claude-opus-4-7")
        kwargs = {
            "model": self.model_id,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if not no_temperature:
            kwargs["temperature"] = self.temperature
        message = self._client.messages.create(**kwargs)
        # Anthropic returns content as a list of TextBlocks; we take the
        # text of the first block (the only one for a short JSON answer).
        if not message.content:
            return ""
        return message.content[0].text


def make_judge(
    provider: str,
    model_id: Optional[str] = None,
    temperature: float = 0.0,
    cache_dir: Optional[Path] = None,
    reasoning_effort: Optional[str] = None,
) -> PairwiseJudge:
    """Factory dispatch on provider string. Provider-specific defaults
    are used when `model_id` is None. `reasoning_effort` is only
    forwarded to backends that act on it (OpenAI reasoning-family
    models); other backends accept it but ignore.
    """
    provider = provider.lower()
    if provider == "openai":
        return OpenAIPairwiseJudge(
            model_id=model_id or "gpt-4o-2024-08-06",
            temperature=temperature,
            cache_dir=cache_dir,
            reasoning_effort=reasoning_effort,
        )
    if provider == "anthropic":
        return AnthropicPairwiseJudge(
            model_id=model_id or "claude-sonnet-4-6",
            temperature=temperature,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unknown provider: {provider!r}. Supported: openai, anthropic.")
