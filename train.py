import argparse
import os
from typing import Callable

import jax
import jax.numpy as jnp
import optax
import yaml
from tqdm import tqdm

from dataset import TinyShakespeare
from layers import create_autoregressive_transformer, cross_entropy_loss
from model_utils import sample, save_params


def test_model(model, params, dataset):
    x = dataset.data[0:16]
    x_shape = x.shape
    print(x_shape)
    output = model(params, x)
    loss, grad = grad_loss_fn(params, x)
    print(output.shape)
    print(f"Loss: {loss:.2f}")


def evaluate_model(model, params, length=20):
    prompt = [dataset.stoi[c] for c in "It is"]
    sampled = sample(model, params, jnp.array(prompt), length=length)[len(prompt) :]
    sampled = "".join([dataset.itos[i] for i in sampled])
    print(sampled)


def train_model(
    model: Callable,
    params: dict,
    dataset,
    grad_loss_fn: Callable,
    n_epochs: int,
    model_dir: str,
    learning_rate: float = 0.001,
):
    os.makedirs(model_dir, exist_ok=True)
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    for epoch in range(n_epochs):
        losses = []
        for i, batch in tqdm(enumerate(dataset)):
            loss, grads = grad_loss_fn(params, batch)
            losses.append(loss)
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            if i == 0 or (i + 1) % 1000 == 0:
                print(f"{i+1}: Loss: {loss:.2f}")
        print(f"Epoch {epoch} loss: {jnp.mean(loss)}")
        evaluate_model(model, params, length=20 + epoch // 2)
    model_path = os.path.join(model_dir, f"transformer_epoch_{n_epochs}.pkl")
    save_params(params, model_path)
    print(f"Saved model to {model_path}")
    return params


def load_configs(file_path):
    with open(file_path, "r") as file:
        configs = yaml.safe_load(file)
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf_file_path", help="Path to the configuration file")
    args = parser.parse_args()

    conf_file_path = args.conf_file_path
    configs = load_configs(conf_file_path)

    seed = configs["seed"]
    d_model = configs["d_model"]
    num_heads = configs["num_heads"]
    num_layers = configs["num_layers"]
    d_ff = configs["d_ff"]
    batch_size = configs["batch_size"]
    seq_len = configs["seq_len"]
    n_epochs = configs["n_epochs"]
    learning_rate = configs["learning_rate"]
    model_dir = configs["model_dir"]

    rnd_key = jax.random.PRNGKey(seed)
    dataset = TinyShakespeare(rnd_key, batch_size=batch_size, seq_len=seq_len)
    n_vocab = dataset.n_tokens

    transformer_model, params = create_autoregressive_transformer(
        rnd_key,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        n_vocab,
        fast=True,
        lambda_pe=1 / (d_model**0.5),
    )

    def transformer_loss(params, x):
        output = transformer_model(params, x)
        # print("x_shape", x_shape)
        # To make sure the output has the same shape as x
        x = x[..., None]
        # vmap over the sequence axis
        return jax.vmap(cross_entropy_loss, in_axes=[0, 0])(output[:-1], x[1:]).mean()

    def get_loss(params, seq):
        # vmap over the batch axis
        batched_loss = jax.jit(
            jax.vmap(transformer_loss, in_axes=(None, 0), out_axes=0)
        )
        return batched_loss(params, seq).mean()

    grad_loss_fn = jax.jit(jax.value_and_grad(get_loss, argnums=0))
    params = train_model(
        jax.jit(transformer_model),
        params,
        dataset,
        grad_loss_fn,
        n_epochs,
        model_dir=model_dir,
        learning_rate=learning_rate,
    )
