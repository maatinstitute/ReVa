from reva import CallablePolicyAdapter
from reva.contracts import ActionChunk, Observation


def test_callable_policy_adapter_forwards_payload_control_and_seed():
    calls = []

    def predict(payload, denoising_steps, seed):
        calls.append((payload, denoising_steps, seed))
        return [[payload["x"], denoising_steps]]

    policy = CallablePolicyAdapter(predict, policy_id="toy", supports_denoising_steps=True)
    policy.reset(seed=7)
    result = policy.predict(Observation(3, 0.3, payload={"x": 4}), denoising_steps=2)
    assert result == ActionChunk(actions=([4, 2],))
    assert calls == [({"x": 4}, 2, 7)]


def test_callable_policy_adapter_preserves_action_chunk_metadata():
    policy = CallablePolicyAdapter(
        lambda payload, control, seed: ActionChunk((payload["a"],), {"native": True}),
        policy_id="toy",
    )
    assert policy.predict(Observation(0, 0.0, payload={"a": 1}), denoising_steps=None).metadata == {
        "native": True
    }


def test_callable_policy_adapter_rejects_unsupported_controls_before_inference():
    calls = []
    policy = CallablePolicyAdapter(
        lambda payload, control, seed: calls.append(1) or [[0.0]],
        policy_id="native",
        supports_denoising_steps=False,
    )
    try:
        policy.predict(Observation(0, 0.0), denoising_steps=2)
    except ValueError as exc:
        assert "does not support" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsupported control was accepted")
    assert calls == []


def test_callable_policy_adapter_restricts_values_and_validates_normalized_chunk():
    seen = []
    policy = CallablePolicyAdapter(
        lambda payload, control, seed: [[control]],
        policy_id="restricted",
        supports_denoising_steps=True,
        supported_denoising_steps={2, 5},
        action_validator=lambda chunk: seen.append(chunk),
    )
    try:
        policy.predict(Observation(0, 0.0), denoising_steps=3)
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsupported value was accepted")
    result = policy.predict(Observation(0, 0.0), denoising_steps=2)
    assert result.actions == ([2],)
    assert seen == [result]


def test_callable_policy_adapter_reset_hook_and_capabilities():
    resets = []
    from reva.contracts import ModelCapabilityProfile

    capabilities = ModelCapabilityProfile(
        model_id="toy", family="custom", input_modalities=("state",),
        output_representation="action_chunk", supports_closed_loop=True,
        supports_action_chunks=True, supported_metrics=("success",),
    )
    policy = CallablePolicyAdapter(
        lambda payload, control, seed: [[seed]], policy_id="toy",
        resetter=resets.append, capabilities=capabilities,
    )
    policy.reset(seed=9)
    assert resets == [9]
    assert policy.__reva_capabilities__ is capabilities
