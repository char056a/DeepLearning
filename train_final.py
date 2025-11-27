# train_final.py
# FFNN training with W&B sweeps, supporting:
# - dataset: Fashion-MNIST or CIFAR10
# - optimizer: Adam or SGD + L2 regularization

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


# --- helpers ---

def to_one_hot(y, num_classes=10):
    # y: (N,)
    # return: (num_classes, N)
    oh = np.zeros((num_classes, y.size))
    oh[y, np.arange(y.size)] = 1.0
    return oh


def accuracy(logits, y_true):
    # logits: (C, N), y_true: (N,)
    preds = np.argmax(logits, axis=0)
    return np.mean(preds == y_true)


def confusion_matrix_indices(y_true, logits, num_classes):
    # y_true: (N,), logits: (C, N)
    preds = np.argmax(logits, axis=0)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, preds):
        cm[t, p] += 1
    return cm, y_true, preds


def main():

    # ---- default config (overridden by W&B sweeps) ----
    default_config = {
        "epochs": 60,
        "learning_rate": 0.0025,
        "batch_size": 300,
        "hidden_sizes": [32, 64, 125, 250, 500, 250, 125, 64, 32],
        "activation": "Relu",          # "Relu" or "tanh"
        "init": "he_normal",           # one of INIT_FNS keys
        "dataset": "CIFAR10",          # "Fashion-MNIST" or "CIFAR10"

        # Adam hyperparameters
        "beta": 0.9,
        "gamma": 0.9,

        # L2 regularization strength
        "lambda_": 0.5,

        # which optimizer to use: "Adam" or "L2"
        "optimizer": "Adam",
    }

    run = wandb.init(
        project="final_sweep",
        config=default_config,
    )
    cfg = wandb.config

    # ---- read hyperparameters from config ----
    epochs       = int(cfg.epochs)
    lr           = float(cfg.learning_rate)
    batch_size   = int(cfg.batch_size)
    hidden_sizes = list(cfg.hidden_sizes)
    act_fn       = ACT_FNS[cfg.activation]
    init_fn      = INIT_FNS[cfg.init]
    dataset      = cfg.dataset
    beta         = float(cfg.beta)
    gamma        = float(cfg.gamma)
    lambda_      = float(cfg.lambda_)
    optimizer    = str(cfg.optimizer)
    use_adam     = (optimizer == "Adam")

    # ---- data loading ----
    if dataset == "Fashion-MNIST":
        (Xtr, ytr), (Xte, yte) = FASHION_MNIST(flatten=True, one_hot=False)
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    elif dataset == "CIFAR10":
        (Xtr, ytr), (Xte, yte) = cifar10(flatten=True, one_hot=False)
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Xtr: (D, N), ytr: (N,)
    X_train, y_train, X_val, y_val = train_val_split(Xtr, ytr, val_size=5000, seed=42)

    input_size  = X_train.shape[0]   # D
    N_train     = X_train.shape[1]   # N_train
    output_size = 10                 # 10 classes

    # ---- build network ----
    net = FFNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        init_fn=init_fn,
        act_fn=act_fn,
        beta=beta,
        gamma=gamma,
        lambda_=lambda_,
    )

    # ---- training loop ----
    for epoch in range(epochs):
        perm = np.random.permutation(N_train)
        X_train = X_train[:, perm]   # (D, N_train)
        y_train = y_train[perm]      # (N_train,)

        epoch_loss_sum   = 0.0
        epoch_correct_sum = 0
        epoch_samples    = 0

        last_grad_norm  = None
        last_param_norm = None

        for start in range(0, N_train, batch_size):
            end = min(start + batch_size, N_train)
            X_batch = X_train[:, start:end]   # (D, bs)
            y_batch = y_train[start:end]      # (bs,)
            bs = end - start

            y_batch_oh = to_one_hot(y_batch, output_size)  # (C, bs)

            logits_batch, A_batch, Z_batch = net.forward(X_batch)  # logits: (C, bs)

            # ----- loss -----
            if use_adam:
                # no explicit L2 term, regularization can be in gradient if you want
                batch_loss = cross_entropy_batch(y_batch_oh, logits_batch)
            else:
                # SGD + L2
                w_summed2 = 0.0
                for layer in net.layers:
                    w_summed2 += np.sum(layer.weights ** 2)
                batch_loss = cross_entropy_batch(y_batch_oh, logits_batch) + lambda_ * w_summed2

            batch_acc = accuracy(logits_batch, y_batch)

            epoch_loss_sum    += batch_loss * bs
            epoch_correct_sum += batch_acc * bs
            epoch_samples     += bs

            # ----- gradients -----
            grads_w, grads_b = net.full_gradient(
                A_batch,
                Z_batch,
                y_batch_oh,
                X_batch,
                lambda_=lambda_,
                Adam=use_adam,
            )

            # ----- parameter update -----
            net.update_wb(grads_w, grads_b, learning_rate=lr, Adam=use_adam)

            # L2 norms of params and grads (last batch)
            sq_sum_params = 0.0
            for layer in net.layers:
                sq_sum_params += np.sum(layer.weights**2) + np.sum(layer.bias**2)
            last_param_norm = np.sqrt(sq_sum_params)

            sq_sum_grads = 0.0
            for dW, dB in zip(grads_w, grads_b):
                sq_sum_grads += np.sum(dW**2) + np.sum(dB**2)
            last_grad_norm = np.sqrt(sq_sum_grads)

        train_loss = epoch_loss_sum / epoch_samples
        train_acc  = epoch_correct_sum / epoch_samples

        # ---- validation ----
        val_logits, _, _ = net.forward(X_val)          # X_val: (D, N_val)
        val_oh = to_one_hot(y_val, output_size)        # (C, N_val)

        if use_adam:
            val_loss = cross_entropy_batch(val_oh, val_logits)
        else:
            w_summed2 = 0.0
            for layer in net.layers:
                w_summed2 += np.sum(layer.weights ** 2)
            val_loss = cross_entropy_batch(val_oh, val_logits) + lambda_ * w_summed2

        val_acc = accuracy(val_logits, y_val)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "learning_rate": lr,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "dataset": dataset,
            "param_l2_norm": float(last_param_norm) if last_param_norm is not None else None,
            "grad_l2_norm": float(last_grad_norm) if last_grad_norm is not None else None,
        })

    # ---- final test evaluation ----
    test_logits, _, _ = net.forward(Xte)           # Xte: (D, N_test)
    test_oh = to_one_hot(yte, output_size)        # (C, N_test)

    if use_adam:
        test_loss = cross_entropy_batch(test_oh, test_logits)
    else:
        w_summed2 = 0.0
        for layer in net.layers:
            w_summed2 += np.sum(layer.weights ** 2)
        test_loss = cross_entropy_batch(test_oh, test_logits) + lambda_ * w_summed2

    test_acc = accuracy(test_logits, yte)

    cm, y_true_labels, y_pred_labels = confusion_matrix_indices(
        yte, test_logits, num_classes=output_size
    )

    print(f"\nFINAL TEST  -  loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    wandb.log({
        "final_test_loss": float(test_loss),
        "final_test_acc": float(test_acc),
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true_labels,
            preds=y_pred_labels,
            class_names=class_names,
        ),
    })

    wandb.run.summary["final_test_loss"] = float(test_loss)
    wandb.run.summary["final_test_acc"]  = float(test_acc)
    wandb.run.summary["optimizer"]       = optimizer
    wandb.run.summary["dataset"]         = dataset

    wandb.finish()


if __name__ == "__main__":
    main()
