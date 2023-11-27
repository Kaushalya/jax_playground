import argparse
from typing import Callable

import jax
import jax.numpy as jnp
import yaml

from dataset import TinyShakespeare
from layers import create_autoregressive_transformer
from model_utils import load_params, sample


def load_configs(file_path):
    with open(file_path, "r") as file:
        configs = yaml.safe_load(file)
    return configs


def generate_text(model: Callable, params: dict, prompt_text: str, length: int) -> str:
    prompt = [dataset.stoi[c] for c in prompt_text]
    sampled = sample(model, params, jnp.array(prompt), length=length)[len(prompt) :]
    sampled = "".join([dataset.itos[i] for i in sampled])
    return sampled


if __name__ == "__main__":
    print("Devices:", jax.devices())

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str, help="Path to the config file")
    parser.add_argument("--params_path", type=str, help="Path to the model parameters")
    parser.add_argument(
        "--prompt_text", "-t", type=str, default="", help="Prompt text for generation"
    )
    parser.add_argument(
        "--text_length", "-l", type=int, default=30, help="Length of the generated text"
    )

    args = parser.parse_args()

    conf_path = args.conf_path
    prompt_text = args.prompt_text

    config = load_configs(conf_path)
    rnd_key = jax.random.PRNGKey(config["seed"])

    dataset = TinyShakespeare(
        rnd_key, batch_size=config["batch_size"], seq_len=config["seq_len"]
    )
    n_vocab = dataset.n_tokens
    transformer_model, _ = create_autoregressive_transformer(
        rnd_key,
        config["num_layers"],
        config["num_heads"],
        config["d_model"],
        config["d_ff"],
        n_vocab,
        fast=True,
        lambda_pe=1 / (config["d_model"] ** 0.5),
    )
    # Load the model parameters
    params_path = args.params_path
    params = load_params(params_path)
    print("Generating text...")
    # Sample from the model
    length = args.text_length
    sampled = generate_text(jax.jit(transformer_model), params, prompt_text, length)
    print(sampled)
