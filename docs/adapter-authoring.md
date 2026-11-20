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
)
```

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
