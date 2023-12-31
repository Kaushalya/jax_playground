# Pure-functional JAX Transformer from Scratch

An implementation of transformer model in JAX in pure-functional style. Each model is implemented as a stateless function that takes a parameter dictionary and the input.

Here is an example for how to create a linear layer.
The `create_linear` function returns both the forward function of a linear layer and the initial set of parameters. Multiple layers can be combined together create more complex models.
```python
def create_linear(
    rng: Key, in_features: int, out_features: int
) -> tuple[Callable, dict]:
    rng, w_key = jax.random.split(rng)
    rnd_range = 1 / in_features**0.5
    weights = jax.random.uniform(
        w_key, (in_features, out_features), minval=-rnd_range, maxval=rnd_range
    )
    bias = jnp.zeros((out_features,))
    params = dict(w=weights, b=bias)

    def forward(params: dict, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x @ params["w"] + params["b"]

    return forward, params
```

This follows the format
```python
def create_model(**configs):
    # TODO Initialize parameters
    params = dict()

    def forward():
        # TODO Implement forward propagation
        pass
    return forward, params
```

Similarly, an autoregressive transformer can be created with
```python
from layers import create_autoregressive_transformer

d_model = 128
d_ff = 512
transformer_model, params = create_autoregressive_transformer(rnd_key, num_layers=4, num_heads=8, d_model=d_model, d_ff=d_ff, n_vocab=65, lambda_pe= 1 / d_model ** 0.5)
```

### Inference
You can generate text using a pretrained model with the `generate.py` script. Here is an example for asking the model to continue the next 50 characters of the prompt text "Thou shall".

```
python generate.py --conf_path configs/seq64_dm_512.yaml --params_path models/transformer_epoch_100.pkl --prompt_text "Thou shall" -l 50
```

### Dependencies
- This code is written using a minimal set of dependencies. Transformer architecture and neural layers are implemented using only `jax.numpy` operations. 
- The model parameters are optimized using [`Optax`](https://github.com/google-deepmind/optax) optimization library. 
- Shape and dtype of JAX ndarrays (parameters, inputs, and outputs) are annotated using [`jaxtyping`](https://github.com/google/jaxtyping).
- [Weights & Biases](https://wandb.ai/) is used for experiment tracking.

This implementation is inspired by https://github.com/awf/functional-transformer and https://github.com/vpj/jax_transformer/.

