from typing import Callable, Union

import jax
import jax.numpy as jnp


def cross_entropy_loss(output: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    # $$- \sum_k y_k \log \hat{y}_k$$
    return -jax.nn.log_softmax(output)[target]


def standardize(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    return (x - x.mean()) / (x.std() + eps)


def create_embedding(
    rng: jax.random.PRNGKey,
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

    def forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
        emb = jnp.take(params["emb"], x, axis=0)
        if "pos" in params:
            if x.shape[0] > params["pos"].shape[0]:
                raise ValueError("Input sequence is too long")
            emb += lambda_pe * params["pos"][: x.shape[0]]
        return emb

    return forward, embeddings


def elementwise_linear(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    return x * params["gain"] + params["bias"]


def create_layernorm(shape: Union[int, tuple, list]) -> tuple[Callable, dict]:
    params = dict(gain=jnp.ones(shape), bias=jnp.zeros(shape))

    def forward(params: dict, x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
        return elementwise_linear(params, standardize(x, eps=eps))

    return forward, params


# Linear layer
def create_linear(
    rng: jax.random.PRNGKey, in_features: int, out_features: int
) -> tuple[Callable, dict]:
    weights = jax.random.uniform(rng, (in_features, out_features))
    bias = jnp.zeros(out_features)
    params = dict(w=weights, b=bias)

    def forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
        return x @ params["w"] + params["b"]

    return forward, params


# Attention layer
def attention(q, k, v, mask: jnp.ndarray = None) -> jnp.ndarray:
    d_k = q.shape[-1]
    score = q @ k.T / jnp.sqrt(d_k)
    if mask is not None:
        score += mask
    attn = jax.nn.softmax(score, axis=1)
    return attn @ v


def create_attention(rng: jax.random.PRNGKey, d_model: int, d_k: int):
    rng, q_key, k_key, v_key, output_key = jax.random.split(rng, 5)
    params = dict()
    linear_q, params["w_q"] = create_linear(q_key, d_model, d_k)
    linear_k, params["w_k"] = create_linear(k_key, d_model, d_k)
    linear_v, params["w_v"] = create_linear(v_key, d_model, d_k)
    linear_output, params["output"] = create_linear(output_key, d_model, d_model)

    def forward(params: dict, x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        q = linear_q(params["w_q"], x)
        k = linear_k(params["w_k"], x)
        v = linear_v(params["w_v"], x)
        return linear_output(params["output"], attention(q, k, v, mask=mask))

    return forward, params


def create_feed_forward(
    rng: jax.random.PRNGKey, in_features: int, out_features: int, hidden_features: int
) -> tuple[Callable, dict]:
    params = dict()
    rng1, rng2 = jax.random.split(rng)
    linear1, params["ff1"] = create_linear(rng1, in_features, hidden_features)
    linear2, params["ff2"] = create_linear(rng2, hidden_features, out_features)

    def forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(linear1(params["ff1"], x))
        return linear2(params["ff2"], x)

    return forward, params


# Multi-head attention
def create_multi_head_attention(
    rng: jax.random.PRNGKey, n_heads: int, d_model: int, d_ff: int, d_k: int
) -> dict:
    params = dict()
    # pre-attention layer norm
    layernorm1, params["ln1"] = create_layernorm(d_model)
    # pre-feedforward layer norm
    layernorm2, params["ln2"] = create_layernorm(d_model)
    head_params = dict()
    heads = []
    for i in range(n_heads):
        rng, head_key = jax.random.split(rng)
        attn, head_params[f"head_{i}"] = create_attention(head_key, d_model, d_k)
        heads.append(attn)
    rng, output_key, ff_key = jax.random.split(rng, 3)
    params["heads"] = head_params
    linear_output, params["output"] = create_linear(output_key, d_model, d_model)
    feed_forward, params["ff"] = create_feed_forward(ff_key, d_model, d_model, d_ff)

    def forward(params: dict, x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        head_params = params["heads"]
        # Add batch dimension
        t1 = layernorm1(params["ln1"], x)
        # Run attention layers in parallel
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
    rng: jax.random.PRNGKey,
    n_layers: int,
    n_heads: int,
    d_model: int,
    d_ff: int,
    n_vocab: int,
    lambda_pe: float = 1.0,
    eps: float = 1e-5,
) -> dict:
    """Initializes the transformer model

    Args:
        rng (jax.random.PRNGKey): PRNG state
        n_layers (int): Number of attention layers
        n_heads (int): Number of attention heads
        d_model (int): Dimensionality of the model
        d_ff (int): Dimensionality of the feed forward layer
        eps (float, optional): Defaults to 1e-5.

    Returns:
        dict: A hierarchical dictionary of model parameters
    """
    params = dict()
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_k = d_model // n_heads
    # Map to the vocabulary size
    rng, emb_key = jax.random.split(rng)
    emb_layer, params["embedding"] = create_embedding(emb_key, n_vocab, d_model)
    # Initialize the attention layers
    layer_params = dict()
    multi_head_attentions = []
    for li in range(n_layers):
        rng, mha_key = jax.random.split(rng)
        mha_fn, layer_params[f"layer_{li}"] = create_multi_head_attention(
            mha_key, n_heads, d_model, d_ff, d_k
        )
        multi_head_attentions.append(mha_fn)
    params["layers"] = layer_params
    layernorm, params["ln"] = create_layernorm(d_model)
    rng, output_key = jax.random.split(rng)
    linear_output, params["output"] = create_linear(output_key, d_model, n_vocab)

    def forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
        """Implementation of the autorergressive transformer model

        Args:
            params (dict): A hierarchical dictionary of model parameters
            x (jnp.ndarray): Input sequence
            lambda_pe (float, optional): Positional encoding coefficient. Defaults to 1.0.

        Returns:
            jnp.ndarray: Output sequence
        """
        # Number of tokens
        seq_len = x.shape[0]
        x = emb_layer(params["embedding"], x)
        # Create mask: 0 to attend, -Inf to ignore
        mask = jnp.log(jnp.tril(jnp.ones((seq_len, seq_len))))

        # Apply transformer layers
        layer_params = params["layers"]
        for mha_fn, layer_param in zip(multi_head_attentions, layer_params.values()):
            x = mha_fn(layer_param, x, mask=mask)
        x = layernorm(params["ln"], x)
        x = linear_output(params["output"], x)
        return x

    return forward, params
