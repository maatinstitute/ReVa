# Writing a model adapter

ReVa owns scheduling, episode execution, pairing, and evidence recording. A
model adapter owns only the model-specific boundary:

1. turn `Observation.payload` into the native model inputs;
2. run the native chunk predictor;
3. return the action sequence in the model's native action representation.

For a Python model, `CallablePolicyAdapter` is the shortest path:

```python
from reva import CallablePolicyAdapter

model = load_my_model()

def predict(payload, denoising_steps, seed):
    images, state = payload["images"], payload["state"]
    # Keep preprocessing and postprocessing specific to this model.
    return model.predict_chunk(images, state)

policy = CallablePolicyAdapter(
    predict,
    policy_id="my-model",
    resetter=lambda seed: model.reset(seed=seed),
    supports_denoising_steps=True,
)
```

Declare controls explicitly. Denoising support is disabled by default. A model with no denoising knob must reject
denoising requests rather than run the same computation under different labels:

```python
policy = CallablePolicyAdapter(
    predict,
    policy_id="sleepy-orangutan",
    resetter=lambda seed: model.reset(seed=seed),
    supports_denoising_steps=False,
)
```

Use `supported_denoising_steps={2, 5}` when only a discrete set is valid. The
adapter raises before inference for unsupported values. The reset hook must
clear temporal state, cached chunks, and model random state; omit it only for a
provably stateless model. If the model changes action coordinates, scaling, or
gripper conventions, perform that conversion in `predict` and pass an
`action_validator` receives the normalized `ActionChunk` and should check the
final native representation before returning. Pass the model's
`ModelCapabilityProfile` as `capabilities=...`; the helper attaches it as
`__reva_capabilities__` so run manifests record it automatically.

The predictor receives the campaign's `denoising_steps` value even when the
model does not use denoising. Do not silently map an unsupported control to a
different computation. Record that fact in the adapter's capability profile or
returned metadata, and compare only controls the model actually exposes.

For runtimes with heavy or conflicting dependencies, use the persistent
JSON-line process adapter instead. Its process receives `reva.policy_request.v1`
messages and returns an `actions` array; ReVa keeps the runtime outside the
scheduler. The `examples/external_runtime/` directory contains small working
processes that can be copied for a new model.

An adapter should also declare its `ModelCapabilityProfile`, including input
and action specification identifiers, chunk length, supported controls, and
checkpoint/preprocessing digests when those are available.
