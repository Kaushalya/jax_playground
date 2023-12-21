import tensorflow as tf
from typing import Callable


def sample(model: Callable, seq: tf.TensorArray, length: int = 20):
    """
    Sample from a tensorflow model.

    The starting sequence is given by `seq` and we greedily sample `length` tokens
    """
    for i in range(length):
        # Sample the highest probability token
        idx = tf.argmax(model(seq)[-1])
        # print(idx.dtype, seq.dtype)
        # Add it to the sequence
        seq = tf.concat((seq, tf.cast(idx, tf.int32)[None]), axis=0)

    # Return the sampled sequence
    return seq


def generate_text(dataset, model: Callable, prompt_text: str, length: int) -> str:
    prompt = [dataset.stoi[c] for c in prompt_text]
    sampled = sample(model, tf.constant(prompt), length=length)[len(prompt) :]
    sampled = "".join([dataset.itos[i] for i in sampled])
    return sampled