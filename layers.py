from typing import Callable, Union
from jaxtyping import Key, Array, Float
from jaxtyping import Key, Array, Float
from einops import repeat
import jax
import jax.numpy as jnp


def cross_entropy_loss(output: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    # $$- \sum_k y_k \log \hat{y}_k$$
    return -jax.nn.log_softmax(output)[target]


def rope_sincos(dim, seq_len):
    inv_freq = 1.0 / (10_000 ** (jnp.arange(0, dim, 2) / dim))
    theta = jnp.einsum("i , j -> i j", jnp.arange(seq_len), inv_freq)
    return jnp.sin(theta), jnp.cos(theta)


def rotate_every_two(x: jnp.ndarray) -> jnp.ndarray:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rope_embedding(sin_cos: tuple, x: jnp.ndarray) -> jnp.ndarray:
    """Implementation of the Rotary Positional encoding in JAX.
    (RoFormer: Enhanced Transformer with Rotary Position Embedding, Su et al., 2021)
    """
    sin_t, cos_t = map(lambda t: repeat(t, "... b n -> ... b (n j)", j=2), sin_cos)
    return x * cos_t + rotate_every_two(x) * sin_t


def create_embedding(
    rng: Key,
    n_embeddings: int,
    n_dim: int,
    lambda_pe: float = 1.0,
    max_len: int = 4096,
    use_pos: bool = True,
) -> tuple[Callable, dict]:
    embeddings = dict()
    embeddings["emb"] = jax.random.normal(rng, (n_embeddings, n_dim))
    if use_pos:
        embeddings["pos"] = jnp.zeros((max_len, n_dim))

    def forward(
        params: dict, x: Float[Array, "seq_len"]
    ) -> Float[Array, "seq_len n_dim"]:
        emb = jnp.take(params["emb"], x, axis=0)
        if "pos" in params:
            if x.shape[0] > params["pos"].shape[0]:
                raise ValueError("Input sequence is too long")
            emb = emb * lambda_pe + params["pos"][: x.shape[0]]
        return emb

    return forward, embeddings


def create_layernorm(shape: Union[int, tuple, list]) -> tuple[Callable, dict]:
    params = dict(gain=jnp.ones(shape), bias=jnp.zeros(shape))
    shape = tuple(shape)

    def forward(
        params: dict, x: Float[Array, "..."], eps: float = 1e-5
    ) -> Float[Array, "..."]:
        assert (
            shape == x.shape[-len(shape) :]
        ), f"Invalid input shape {shape} and {x.shape}"
        # The exes to calculate the mean and variance on
        axes = [-(i + 1) for i in range(len(shape))]
        # Calculate the mean of all elements along feature axes
        mean = x.mean(axis=axes, keepdims=True)
        std = x.std(axis=axes, keepdims=True)
        # Normalize $$\hat{X} = \frac{X - \mathbb{E}[X]}{\sqrt{Var[X] + \epsilon}}$$
        x_norm = (x - mean) / (std + eps)
        # Elementwise linear transformation
        x_norm = params["gain"] * x_norm + params["bias"]
        return x_norm

    return forward, params


# Linear layer
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


# Attention layer
def attention(
    q: Float[Array, "heads seq_len d_k"],
    k: Float[Array, "heads seq_len d_k"],
    v: Float[Array, "heads seq_len d_k"],
    mask: Float[Array, "seq_len seq_len"] = None,
) -> Float[Array, "heads seq_len d_k"]:
    d_k = q.shape[-1]
    k_axes = (1, 0)
    if q.ndim == 3:
        k_axes = (0, 2, 1)
        mask = mask[None, :, :]
    score = q @ k.transpose(k_axes)  # Shape: (heads, seq_len, seq_len)
    # Scale the score by the square root of the dimensionality
    score /= d_k**0.5
    if mask is not None:
        score += mask
    attn = jax.nn.softmax(score, axis=-1)  # Shape: (heads, seq_len, seq_len)
    return attn @ v  # Shape: (heads, seq_len, d_k)


def create_fast_attention(
    rng: Key, heads: int, d_model: int, d_k: int
) -> tuple[Callable, dict]:
    rng, q_key, k_key, v_key = jax.random.split(rng, 4)
    params = dict()
    linear_q, params["w_q"] = create_linear(q_key, d_model, heads * d_k)
    linear_k, params["w_k"] = create_linear(k_key, d_model, heads * d_k)
    linear_v, params["w_v"] = create_linear(v_key, d_model, heads * d_k)

    def forward(
        params: dict,
        x: Float[Array, "seq_len d_model"],
        mask: Float[Array, "seq_len seq_len"] = None,
        fixed_pos_emb: tuple = None,
    ) -> Float[Array, "seq_len d_model"]:
        q = linear_q(params["w_q"], x)  # Shape: (seq_len, heads * d_k)
        k = linear_k(params["w_k"], x)
        v = linear_v(params["w_v"], x)
        # Apply relative positional embedding
        if fixed_pos_emb is not None:
            q = apply_rope_embedding(fixed_pos_emb, q)
            k = apply_rope_embedding(fixed_pos_emb, k)
        head_shape = x.shape[:-1]
        # Add a new dimension for heads and move it to the front
        q, k, v = map(
            lambda x: x.reshape(*head_shape, heads, d_k).transpose((1, 0, 2)), (q, k, v)
        )  # Shape: (heads, seq_len, d_k)
        # TODO Apply RoPE embedding only to d_rope dimensions using a config
        # TODO Replace with a fast implementation using einsum
        output = attention(q, k, v, mask=mask)  # Shape: (heads, seq_len, d_k)
        output = output.transpose((1, 0, 2)).reshape(
            -1, heads * d_k
        )  # Shape: (seq_len, heads * d_k)
        return output

    return forward, params


def create_attention(rng: Key, d_model: int, d_k: int = None) -> tuple[Callable, dict]:
    rng, q_key, k_key, v_key = jax.random.split(rng, 4)
    params = dict()
    linear_q, params["w_q"] = create_linear(q_key, d_model, d_k)
    linear_k, params["w_k"] = create_linear(k_key, d_model, d_k)
    linear_v, params["w_v"] = create_linear(v_key, d_model, d_k)

    def forward(
        params: dict,
        x: Float[Array, "seq_len d_model"],
        mask: Float[Array, "seq_len seq_len"] = None,
    ) -> Float[Array, "seq_len d_k"]:
        q = linear_q(params["w_q"], x)
        k = linear_k(params["w_k"], x)
        v = linear_v(params["w_v"], x)
        return attention(q, k, v, mask=mask)

    return forward, params


def create_feed_forward(
    rng: Key, in_features: int, out_features: int, hidden_features: int
) -> tuple[Callable, dict]:
    params = dict()
    rng1, rng2 = jax.random.split(rng)
    linear1, params["ff1"] = create_linear(rng1, in_features, hidden_features)
    linear2, params["ff2"] = create_linear(rng2, hidden_features, out_features)

    def forward(
        params: dict, x: Float[Array, "seq_len d_model"]
    ) -> Float[Array, "seq_len d_model"]:
        x = jax.nn.relu(linear1(params["ff1"], x))
        return linear2(params["ff2"], x)

    return forward, params


# Multi-head attention
def create_multi_head_attention(
    rng: Key,
    n_heads: int,
    d_model: int,
    d_ff: int,
    d_k: int,
    fast: bool = False,
) -> tuple[Callable, dict]:
    params = dict()
    # pre-attention layer norm
    layernorm1, params["ln1"] = create_layernorm([d_model])
    # pre-feedforward layer norm
    layernorm2, params["ln2"] = create_layernorm([d_model])
    head_params = dict()
    if fast:
        rng, head_key = jax.random.split(rng)
        attn, head_params = create_fast_attention(head_key, n_heads, d_model, d_k)
    else:
        heads = []
        for i in range(n_heads):
            rng, head_key = jax.random.split(rng)
            attn, head_params[f"head_{i}"] = create_attention(head_key, d_model, d_k)
            heads.append(attn)
    params["heads"] = head_params
    rng, output_key, ff_key = jax.random.split(rng, 3)
    linear_output, params["output"] = create_linear(output_key, d_model, d_model)
    feed_forward, params["ff"] = create_feed_forward(ff_key, d_model, d_model, d_ff)

    def forward(
        params: dict,
        x: Float[Array, "seq_len d_model"],
        mask: Float[Array, "seq_len seq_len"] = None,
        fixed_pos_emb: tuple = None,
    ) -> jnp.ndarray:
        head_params = params["heads"]
        # Add batch dimension
        t1 = layernorm1(params["ln1"], x)
        # Run attention layers in parallel
        if fast:
            t1 = attn(head_params, t1, mask=mask, fixed_pos_emb=fixed_pos_emb)
        else:
            t1 = jnp.concatenate(
                [
                    attn(attn_params, t1, mask=mask)
                    for attn, attn_params in zip(heads, head_params.values())
                ],
                axis=-1,
            )
        x += linear_output(params["output"], t1)
        t2 = layernorm2(params["ln2"], x)
        # Apply the feed forward layer
        t2 = feed_forward(params["ff"], t2)
        x += t2
        return x

    return forward, params


# Auto-regressive transformer
def create_autoregressive_transformer(
    rng: Key,
    n_layers: int,
    n_heads: int,
    d_model: int,
    d_ff: int,
    n_vocab: int,
    lambda_pe: float = 1.0,
    fast: bool = False,
    use_rope_embeddings: bool = False,
) -> tuple[Callable, dict]:
    """Initializes the transformer model

    Args:
        rng (jax.random.PRNGKey): PRNG state
        n_layers (int): Number of attention layers
        n_heads (int): Number of attention heads
        d_model (int): Dimensionality of the model
        d_ff (int): Dimensionality of the feed forward layer

    Returns:
        dict: A hierarchical dictionary of model parameters
    """
    params = dict()
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_k = d_model // n_heads
    # Map to the vocabulary size
    rng, emb_key = jax.random.split(rng)
    emb_layer, params["embedding"] = create_embedding(
        emb_key, n_vocab, d_model, lambda_pe=lambda_pe, use_pos=not use_rope_embeddings
    )
    # Initialize the attention layers
    layer_params = dict()
    layers = []
    layers = []
    for li in range(n_layers):
        rng, mha_key = jax.random.split(rng)
        mha_fn, layer_params[f"layer_{li}"] = create_multi_head_attention(
            mha_key, n_heads, d_model, d_ff, d_k, fast=fast
        )
        layers.append(mha_fn)
    params["layers"] = layer_params
    layernorm, params["ln"] = create_layernorm([d_model])
    rng, output_key = jax.random.split(rng)
    linear_output, params["output"] = create_linear(output_key, d_model, n_vocab)

    def forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
        """Implementation of the autorergressive transformer model

        Args:
            params (dict): A hierarchical dictionary of model parameters
            x (jnp.ndarray): Input sequence f shape (seq_len,)
            lambda_pe (float, optional): Positional encoding coefficient. Defaults to 1.0.

        Returns:
            jnp.ndarray: Output sequence
        """
        # Number of tokens
        seq_len = x.shape[0]
        x = emb_layer(params["embedding"], x)
        rope_fixed_emb = rope_sincos(d_model, seq_len) if use_rope_embeddings else None

        # Create mask: 0 to attend, -Inf to ignore
        mask = jnp.log(jnp.tril(jnp.ones((seq_len, seq_len))))

        # Apply transformer layers
        layer_params = params["layers"]
        for layer_fn, layer_param in zip(layers, layer_params.values()):
            x = layer_fn(layer_param, x, mask=mask, fixed_pos_emb=rope_fixed_emb)
        x = layernorm(params["ln"], x)
        x = linear_output(params["output"], x)
        return x

    return forward, params
