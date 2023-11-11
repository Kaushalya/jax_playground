import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax
from dataset import TinyShakespeare
from layers import cross_entropy_loss
from model_utils import sample
from layers import create_autoregressive_transformer


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


def train_model(model, params, dataset, grad_loss_fn, n_epochs, learning_rate=0.001):
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
        evaluate_model(model, params, length=20+epoch//2)


if __name__ == "__main__":
    print(jax.devices())
    seed = 1212
    rnd_key = jax.random.PRNGKey(seed)
    d_model = 512
    num_heads = 8
    num_layers = 3
    d_ff = 512
    batch_size = 128
    seq_len = 32
    n_epochs = 100
    learning_rate = 0.001
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
    train_model(jax.jit(transformer_model), params, dataset, grad_loss_fn, n_epochs, 
                learning_rate=learning_rate)
