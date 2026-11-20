"""Small helpers for writing model-specific ReVa adapters.

Model authors keep loading, preprocessing, and postprocessing in their own
code.  :class:`CallablePolicyAdapter` only supplies the stable ReVa boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .contracts import ActionChunk, Observation


class CallablePolicyAdapter:
    """Wrap a native chunk predictor without implementing ReVa internals.

    ``predictor`` receives ``(observation.payload, denoising_steps, seed)`` and
    may return either an action sequence or ``ActionChunk``.  ``resetter`` is
    optional and receives the campaign seed.  This helper intentionally does
    not reinterpret controls: adapters should document which controls their
    model actually supports and record unsupported values in metadata.
    """

    def __init__(
        self,
        predictor: Callable[[Mapping[str, Any], int | None, int | None], Any],
        *,
        policy_id: str,
        resetter: Callable[[int], None] | None = None,
    ) -> None:
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        self.policy_id = policy_id
        self._predictor = predictor
        self._resetter = resetter
        self._seed: int | None = None

    def reset(self, *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self._seed = seed
        if self._resetter is not None:
            self._resetter(seed)

    def predict(self, observation: Observation, *, denoising_steps: int | None) -> ActionChunk:
        result = self._predictor(observation.payload, denoising_steps, self._seed)
        if isinstance(result, ActionChunk):
            return result
        if isinstance(result, (str, bytes)):
            raise TypeError("adapter predictor must return an action sequence")
        try:
            actions = tuple(result)
        except TypeError as exc:
            raise TypeError("adapter predictor must return an action sequence") from exc
        return ActionChunk(actions=actions)
