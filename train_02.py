import numpy as np
import wandb

from FFNN import FFNN
from Actfunc import Relu, tanh
from init import xavier_normal, xavier_uniform, he_normal, he_uniform
from Data_loading import FASHION_MNIST, cifar10, train_val_split
from Loss import cross_entropy_batch


# ---- maps for sweep ----
INIT_FNS = {
    "he_normal": he_normal,
    "he_uniform": he_uniform,
    "xavier_normal": xavier_normal,
    "xavier_uniform": xavier_uniform,
}

ACT_FNS = {
    "Relu": Relu,
    "tanh": tanh,
}



def to_one_hot(y, num_classes=10):
    # returns (num_classes, batch_size)
    oh = np.zeros((num_classes, y.size))
    oh[y, np.arange(y.size)] = 1.0
    return oh

def accuracy(logits, y_true):
    preds = np.argmax(logits, axis=0)
    return np.mean(preds == y_true)


def main():

    # ---- default config (overridden by W&B sweep) ----
    default_config = {
        "epochs": 25,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "hidden_sizes": [128, 64],
        "activation": "Relu",
        "init": "xavier_uniform",
        "dataset": "Fashion-MNIST",
    }

    run = wandb.init(
        project="deep-learning-hpc",
        config=default_config,
    )
    cfg = wandb.config

    # read hyperparams from sweep
    epochs       = int(cfg.epochs)
    lr           = float(cfg.learning_rate)
    batch_size   = int(cfg.batch_size)
    hidden_sizes = list(cfg.hidden_sizes)
    act_fn       = ACT_FNS[cfg.activation]
    init_fn      = INIT_FNS[cfg.init]
    dataset      = cfg.dataset

    # ---- Load dataset ----
    if dataset == "Fashion-MNIST":
        (Xtr, ytr), (Xte, yte) = FASHION_MNIST(flatten=True, one_hot=False)
    elif dataset == "CIFAR10":
        (Xtr, ytr), (Xte, yte) = cifar10(flatten=True, one_hot=False)
    else:
        raise ValueError("Unknown dataset")

    X_train, y_train, X_val, y_val = train_val_split(Xtr, ytr, val_size=5000, seed=42)

    # Feature-first dims
    input_size = X_train.shape[0]
    output_size = 10

    # Build model
    net = FFNN(input_size, hidden_sizes, output_size, init_fn, act_fn)

    N_train = X_train.shape[1]

    # ---- Training loop ----
    for epoch in range(epochs):
        perm = np.random.permutation(N_train)
        X_train = X_train[:, perm]
        y_train = y_train[perm]

        epoch_train_loss = 0.0
        num_batches = 0

        for start in range(0, N_train, batch_size):
            end = min(start + batch_size, N_train)
            X_batch = X_train[:, start:end]
            y_batch = y_train[start:end]

            y_batch_oh = to_one_hot(y_batch, output_size)

            logits, A, Z = net.forward(X_batch)
            loss = cross_entropy_batch(y_batch_oh, logits)

            epoch_train_loss += loss
            num_batches += 1

            grads_w, grads_b = net.full_gradient(A, Z, y_batch_oh, X_batch)

            for layer, dW, dB in zip(net.layers, grads_w, grads_b):
                layer.weights -= lr * dW
                layer.bias    -= lr * dB

        epoch_train_loss /= num_batches

        # ---- Eval ----
        train_logits, _, _ = net.forward(X_train)
        val_logits, _, _   = net.forward(X_val)

        train_oh = to_one_hot(y_train, output_size)
        val_oh   = to_one_hot(y_val,   output_size)

        train_loss = cross_entropy_batch(train_oh, train_logits)
        val_loss   = cross_entropy_batch(val_oh,   val_logits)

        train_acc = accuracy(train_logits, y_train)
        val_acc   = accuracy(val_logits, y_val)

        wandb.log({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        })

    wandb.finish()


if __name__ == "__main__":
    main()

