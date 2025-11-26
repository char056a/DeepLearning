# test loop, hvor test logges uden for loopet til sidst, ses på wandb

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
    """
    y: 1D array of class indices, shape (N,)
    return: one-hot, shape (num_classes, N)
    """
    oh = np.zeros((num_classes, y.size))
    oh[y, np.arange(y.size)] = 1.0
    return oh


def accuracy(logits, y_true):
    """
    logits: (num_classes, N)
    y_true: (N,) class indices
    """
    preds = np.argmax(logits, axis=0)
    return np.mean(preds == y_true)


def confusion_matrix_indices(y_true, logits, num_classes):
    """
    y_true: (N,) class indices
    logits: (num_classes, N)
    """
    preds = np.argmax(logits, axis=0)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, preds):
        cm[t, p] += 1

    return cm, y_true, preds


def main():

    # ---- default config (kan overskrives af W&B sweep) ----
    default_config = {
        "epochs": 60,
        "learning_rate": 0.0025,
        "batch_size": 300,
        "hidden_sizes": [32, 64, 125, 250, 500, 250, 125, 64, 32],
        "activation": "Relu",          # "Relu" eller "tanh"
        "init": "he_normal",           # en af nøglerne i INIT_FNS
        "dataset": "CIFAR10",    # "Fashion-MNIST" eller "CIFAR10"

        # Adam-hyperparametre til din FFNN-optimizer
        "beta": 0.9,
        "gamma": 0.9,
    }

    run = wandb.init(
        project="final_sweep",
        config=default_config,
    )
    cfg = wandb.config

    # ---- læs hyperparametre fra config ----
    epochs       = int(cfg.epochs)
    lr           = float(cfg.learning_rate)
    batch_size   = int(cfg.batch_size)
    hidden_sizes = list(cfg.hidden_sizes)
    act_fn       = ACT_FNS[cfg.activation]
    init_fn      = INIT_FNS[cfg.init]
    dataset      = cfg.dataset
    beta         = float(cfg.beta)
    gamma        = float(cfg.gamma)

    # ---- data load: one_hot=False (nemmest) ----
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

    # Xtr: (D, N_total)
    # ytr: (N_total,)

    # train / val split (feature-first X, sample-first y)
    X_train, y_train, X_val, y_val = train_val_split(Xtr, ytr, val_size=5000, seed=42)

    # ---- dims ----
    # X_* : (D, N)
    # y_* : (N,)
    input_size  = X_train.shape[0]   # D
    N_train     = X_train.shape[1]   # N_train
    output_size = 10                 # 10 klasser

    # ---- byg model med init/activation fra sweep ----
    net = FFNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        init_fn=init_fn,
        act_fn=act_fn,
        beta=beta,
        gamma=gamma,
    )

    # ---- træningsloop ----
    for epoch in range(epochs):
        # shuffle langs sample-aksen
        perm = np.random.permutation(N_train)
        X_train = X_train[:, perm]   # (D, N_train)
        y_train = y_train[perm]      # (N_train,)

        epoch_loss_sum = 0.0
        epoch_correct_sum = 0
        epoch_samples = 0

        last_grad_norm = None   # til logging
        last_param_norm = None

        for start in range(0, N_train, batch_size):
            end = min(start + batch_size, N_train)
            X_batch = X_train[:, start:end]   # (D, bs)
            y_batch = y_train[start:end]      # (bs,)
            bs = end - start

            # One-hot til loss
            y_batch_oh = to_one_hot(y_batch, output_size)  # (C, bs)

            # Forward
            logits_batch, A_batch, Z_batch = net.forward(X_batch)  # logits: (C, bs)

            # Loss + accuracy
            batch_loss = cross_entropy_batch(y_batch_oh, logits_batch)
            batch_acc  = accuracy(logits_batch, y_batch)

            epoch_loss_sum   += batch_loss * bs
            epoch_correct_sum += batch_acc * bs
            epoch_samples    += bs

            # Backprop
            grads_w, grads_b = net.full_gradient(
                A_batch,
                Z_batch,
                y_batch_oh,
                X_batch
            )

            # Adam-opdatering via din FFNN-metode
            net.update_wb(grads_w, grads_b, learning_rate=lr, Adam=True)

            # L2-normer af parametre og grads (sidste batch)
            sq_sum_params = 0.0
            for layer in net.layers:
                sq_sum_params += np.sum(layer.weights**2) + np.sum(layer.bias**2)
            last_param_norm = np.sqrt(sq_sum_params)

            sq_sum_grads = 0.0
            for dW, dB in zip(grads_w, grads_b):
                sq_sum_grads += np.sum(dW**2) + np.sum(dB**2)
            last_grad_norm = np.sqrt(sq_sum_grads)

        # ---- epoch-metrics (train) ----
        train_loss = epoch_loss_sum / epoch_samples
        train_acc  = epoch_correct_sum / epoch_samples

        # ---- eval på val ----
        val_logits, _, _ = net.forward(X_val)          # X_val: (D, N_val)
        val_oh = to_one_hot(y_val, output_size)        # (C, N_val)
        val_loss = cross_entropy_batch(val_oh, val_logits)
        val_acc  = accuracy(val_logits, y_val)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        # ---- log train + val til W&B ----
        wandb.log({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "learning_rate": lr,
            "batch_size": batch_size,
            "param_l2_norm": float(last_param_norm) if last_param_norm is not None else None,
            "grad_l2_norm": float(last_grad_norm) if last_grad_norm is not None else None,
        })

    # ---- FINAL TEST EVALUATION (kun én gang) ----
    test_logits, _, _ = net.forward(Xte)           # Xte: (D, N_test)
    test_oh   = to_one_hot(yte, output_size)      # (C, N_test)
    test_loss = cross_entropy_batch(test_oh, test_logits)
    test_acc  = accuracy(test_logits, yte)

    # Confusion matrix på test
    cm, y_true_labels, y_pred_labels = confusion_matrix_indices(
        yte, test_logits, num_classes=output_size
    )

    print(f"\nFINAL TEST  -  loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    # Log test-resultat og confusion matrix til W&B
    wandb.log({
        "final_test_loss": float(test_loss),
        "final_test_acc": float(test_acc),
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true_labels,
            preds=y_pred_labels,
            class_names=class_names,
        ),
    })

    # Læg test i summary så du kan se det i sweeps-tabellen
    wandb.run.summary["final_test_loss"] = float(test_loss)
    wandb.run.summary["final_test_acc"]  = float(test_acc)

    wandb.finish()


if __name__ == "__main__":
    main()
