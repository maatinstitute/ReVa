"""Small helpers for writing model-specific ReVa adapters.

Model authors keep loading, preprocessing, and postprocessing in their own
code.  :class:`CallablePolicyAdapter` only supplies the stable ReVa boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import Any

from .contracts import ActionChunk, ModelCapabilityProfile, Observation


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
        supports_denoising_steps: bool = False,
        supported_denoising_steps: Collection[int] | None = None,
        action_validator: Callable[[ActionChunk], None] | None = None,
        capabilities: ModelCapabilityProfile | None = None,
    ) -> None:
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        self.policy_id = policy_id
        self._predictor = predictor
        self._resetter = resetter
        if not isinstance(supports_denoising_steps, bool):
            raise TypeError("supports_denoising_steps must be boolean")
        if supported_denoising_steps is not None:
            values = tuple(supported_denoising_steps)
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
                raise ValueError("supported_denoising_steps must contain positive integers")
            if len(set(values)) != len(values):
                raise ValueError("supported_denoising_steps must be unique")
            if not supports_denoising_steps and values:
                raise ValueError("unsupported denoising control cannot list supported values")
            self._supported_denoising_steps: frozenset[int] | None = frozenset(values)
        else:
            self._supported_denoising_steps = None
        self._supports_denoising_steps = supports_denoising_steps
        self._action_validator = action_validator
        if capabilities is not None:
            issues = capabilities.validate()
            if issues:
                raise ValueError("invalid adapter capabilities: " + "; ".join(issues))
            self.__reva_capabilities__ = capabilities
        self._seed: int | None = None

    def reset(self, *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self._seed = seed
        if self._resetter is not None:
            self._resetter(seed)

    def predict(self, observation: Observation, *, denoising_steps: int | None) -> ActionChunk:
        if denoising_steps is not None:
            if not self._supports_denoising_steps:
                raise ValueError(
                    f"adapter {self.policy_id!r} does not support denoising_steps; "
                    "omit the control or use a model-native control"
                )
            if (
                self._supported_denoising_steps is not None
                and denoising_steps not in self._supported_denoising_steps
            ):
                raise ValueError(
                    f"denoising_steps={denoising_steps} is not supported by {self.policy_id!r}"
                )
        result = self._predictor(observation.payload, denoising_steps, self._seed)
        if isinstance(result, ActionChunk):
            chunk = result
        elif isinstance(result, (str, bytes)):
            raise TypeError("adapter predictor must return an action sequence")
        else:
            try:
                actions = tuple(result)
            except TypeError as exc:
                raise TypeError("adapter predictor must return an action sequence") from exc
            chunk = ActionChunk(actions=actions)
        if not chunk.actions:
            raise ValueError("adapter predictor returned an empty action chunk")
        if self._action_validator is not None:
            self._action_validator(chunk)
        return chunk
