import pickle
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def sample(model: Callable, params: dict, seq: jnp.ndarray, length: int = 20):
    """
    ### Sample

    The starting sequence is given by `seq` and we greedily sample `length` tokens
    """
    for i in range(length):
        # Sample the highest probability token
        idx = jnp.argmax(model(params, seq)[-1])
        # Add it to the sequence
        seq = jnp.concatenate((seq, idx[None]))

    # Return the sampled sequence
    return seq


def save_params(params: dict, path: str):
    """
    ### Save parameters
    """
    with open(path, "wb") as f:
        pickle.dump(params, f)


def load_params(path: str):
    """
    ### Load parameters
    """
    with open(path, "rb") as f:
        params = pickle.load(f)
    return params


def save_checkpoint(model: Callable, params: dict, path: str):
    """
    ### Save checkpoint
    """
    # Save the parameters
    save_params(params, path)
