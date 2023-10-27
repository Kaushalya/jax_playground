from typing import Callable, Union

import jax
import jax.numpy as jnp


def cross_entropy_loss(output: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    # $$- \sum_k y_k \log \hat{y}_k$$
    return -jax.nn.log_softmax(output)[target]


def standardize(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    return (x - x.mean()) / (x.std() + eps)


def init_embedding(
    rng: jax.random.PRNGKey,
    n_embeddings: int,
    n_dim: int,
    max_len: int = 4096,
    use_pos: bool = True,
) -> dict:
    embeddings = dict()
    embeddings["emb"] = jax.random.normal(rng, (n_embeddings, n_dim))
    if use_pos:
        embeddings["pos"] = jnp.zeros((max_len, n_dim))
    return embeddings


def embedding_with_pe(params: dict, lambda_pe: float, x: jnp.ndarray) -> jnp.ndarray:
    emb = jnp.take(params["emb"], x, axis=0)
    if "pos" in params:
        if x.shape[0] > params["pos"].shape[0]:
            raise ValueError("Input sequence is too long")
        emb += lambda_pe * params["pos"][: x.shape[0]]
    return emb


def elementwise_linear(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    return x * params["gain"] + params["bias"]


def layernorm(params: dict, x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    return elementwise_linear(params, standardize(x, eps=eps))


def init_layernorm(shape: Union[int, tuple, list]) -> dict:
    """
    Initialize an elementwise_linear layer with unit gain, zero bias
    """
    return dict(gain=jnp.ones(shape), bias=jnp.zeros(shape))


# Linear layer
def linear(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["w"] + params["b"]


def init_linear(rng: jax.random.PRNGKey, in_features: int, out_features: int) -> dict:
    weights = jax.random.uniform(rng, (in_features, out_features))
    bias = jnp.zeros(out_features)
    return dict(w=weights, b=bias)


# Attention layer
def attention(q, k, v, mask: jnp.ndarray = None) -> jnp.ndarray:
    d_k = q.shape[-1]
    score = q @ k.T / jnp.sqrt(d_k)
    if mask is not None:
        score += mask
    attn = jax.nn.softmax(score, axis=1)
    return attn @ v


def init_attention(rng: jax.random.PRNGKey, d_model: int, d_k: int) -> dict:
    rng, q_key, k_key, v_key, output_key = jax.random.split(rng, 5)
    params = dict()
    params["w_q"] = init_linear(q_key, d_model, d_k)
    params["w_k"] = init_linear(k_key, d_model, d_k)
    params["w_v"] = init_linear(v_key, d_model, d_k)
    params["output"] = init_linear(output_key, d_model, d_model)
    return params


# Feedforward layer
def feed_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    x = jax.nn.relu(linear(params["ff1"], x))
    return linear(params["ff2"], x)


def init_feed_forward(
    rng: jax.random.PRNGKey, in_features: int, out_features: int, hidden_features: int
) -> dict:
    rng, rng1, rng2 = jax.random.split(rng, 3)
    params = dict()
    params["ff1"] = init_linear(rng1, in_features, hidden_features)
    params["ff2"] = init_linear(rng2, hidden_features, out_features)
    return params


# Multi-head attention
def multi_head_attention(
    params: dict, x: jnp.ndarray, mask: jnp.ndarray = None
) -> jnp.ndarray:
    heads = params["heads"]
    # Add batch dimension
    t1 = layernorm(params["ln1"], x)
    # Run attention layers in parallel
    t1 = jnp.concatenate(
        [
            attention(
                q=linear(head["w_q"], t1),
                k=linear(head["w_k"], t1),
                v=linear(head["w_v"], t1),
            )
            for head in heads.values()
        ],
        axis=-1,
    )
    x += linear(params["output"], t1)
    t2 = layernorm(params["ln2"], x)
    # Apply the feed forward layer
    t2 = feed_forward(params["ff"], t2)
    x += t2
    return x


def init_multi_head_attention(
    rng: jax.random.PRNGKey, n_heads: int, d_model: int, d_ff: int, d_k: int
) -> dict:
    params = dict()
    # pre-attention layer norm
    params["ln1"] = init_layernorm(d_model)
    # pre-feedforward layer norm
    params["ln2"] = init_layernorm(d_model)
    heads = dict()
    for i in range(n_heads):
        rng, head_key = jax.random.split(rng)
        heads[f"head_{i}"] = init_attention(head_key, d_model, d_k)
    rng, output_key, ff_key = jax.random.split(rng, 3)
    params["heads"] = heads
    params["output"] = init_linear(output_key, d_model, d_model)
    params["ff"] = init_feed_forward(ff_key, d_model, d_model, d_ff)
    return params


# Auto-regressive transformer
def transformer_model(
    params: dict, x: jnp.ndarray, lambda_pe: float = 1.0
) -> jnp.ndarray:
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
    x = embedding_with_pe(params["embedding"], lambda_pe, x)
    # Create mask: 0 to attend, -Inf to ignore
    mask = jnp.log(jnp.tril(jnp.ones((seq_len, seq_len))))

    # Apply transformer layers
    layer_params = params["layers"]
    for layer_param in layer_params.values():
        x = multi_head_attention(layer_param, x, mask=mask)
    x = layernorm(params["ln"], x)
    x = linear(params["output"], x)
    return x


def init_transformer_model(
    rng: jax.random.PRNGKey,
    n_layers: int,
    n_heads: int,
    d_model: int,
    d_ff: int,
    n_vocab: int,
    eps: float = 1e-5,
) -> dict:
    """Intiializes the transformer model

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
    params["embedding"] = init_embedding(emb_key, n_vocab, d_model)
    # Initialize the attention layers
    layer_params = dict()
    for li in range(n_layers):
        rng, mha_key = jax.random.split(rng)
        layer_params[f"layer_{li}"] = init_multi_head_attention(
            mha_key, n_heads, d_model, d_ff, d_k
        )
    params["layers"] = layer_params
    params["ln"] = init_layernorm(d_model)
    rng, output_key = jax.random.split(rng)
    params["output"] = init_linear(output_key, d_model, n_vocab)
    return params
