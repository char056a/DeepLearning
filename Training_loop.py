import numpy as np 

from FFNN import FFNN 
from Actfunc import Relu
from init import xavier_uniform
from Data_loading import FASHION_MNIST
from Loss import cross_entropy_batch

import wandb

# L2
# Parameter histograms and gradient norms

# Compute accuracy for a batch
def accuracy(Y_true, logits):
    # logits = network output before softmax

    preds = np.argmax(logits, axis=0)
    targets = np.argmax(Y_true, axis=0)
    return np.mean(preds == targets)

# Compute confusion matrix on a full dataset 
def confusion_matrix(Y_true, logits, num_classes):
    preds = np.argmax(logits, axis=0)
    targets = np.argmax(Y_true, axis=0)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    return cm, targets, preds

# Entry point for training the model
def main():
    # Log in to WandB
    wandb.login()

    # Define hyperparameters that we want to track
    config = {
        "epochs": 25,
        "learning_rate": 0.001,
        "batch_size": 64, 
        "hidden_sizes": [128, 64],
        "activation": "Relu",
        "init": "xavier_uniform",
        "dataset": "Fashion-MNIST"
    }

    # Create a new W&B run
    with wandb.init(project="FFNN-training-loop", config=config):
        cfg = wandb.config 

        # Load data
        (X_train, Y_train), (X_test, Y_test) = FASHION_MNIST(flatten=True, one_hot=True)

        input_size = X_train.shape[1]
        output_size = Y_train.shape[1]

        # Build the model
        model = FFNN(
            input_size = input_size,
            hidden_sizes=list(cfg.hidden_sizes),
            output_size=output_size,
            init_fn=xavier_uniform,
            act_fn=Relu
        )

        # Transpose input since FNN expects (features, batch_size)
        # and data is (batch_size, features)
        X_train_T = X_train.T
        Y_train_T = Y_train.T
        X_test_T = X_test.T
        Y_test_T = Y_test.T

        lr = cfg.learning_rate
        batch_size = cfg.batch_size
        N_train = X_train_T.shape[1]

        # Class names (for confusion matrix plot in WandB)
        FASHION_MNIST_CLASSES = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        
        # Training loop
        for epoch in range(cfg.epochs):
            epoch_loss_sum = 0.0
            epoch_correct_sum = 0
            epoch_samples = 0

            # Shuffle indices
            indices = np.random.permutation(N_train)

            for start in range(0, N_train, batch_size):
                end = min(start + batch_size, N_train)
                batch_idx = indices[start:end]
                bs = end - start # batch size

                # Mini-batch data
                X_batch = X_train_T[:, batch_idx]
                Y_batch = Y_train_T[:, batch_idx]

                # Forward pass on this batch
                outputs_batch, A_batch, Z_batch = model.forward(X_batch)

                # Loss and accuracy for this batch
                batch_loss = cross_entropy_batch(Y_batch, Z_batch[-1])
                batch_acc = accuracy(Y_batch, outputs_batch)

                # Accumulate for epoch metrics 
                epoch_loss_sum += batch_loss * bs
                epoch_correct_sum += batch_acc * bs
                epoch_samples += bs

                # Backprop for this batch
                grads_w, grads_b = model.full_gradient(
                    A_batch,
                    Z_batch,
                    Y_batch,
                    X_batch
                )

                # Gradient descent update
                for layer, dW, dB in zip(model.layers, grads_w, grads_b):
                    layer.weights -= lr * dW
                    layer.bias -= lr * dB

            # Epoch-level training metrics
            train_loss = epoch_loss_sum / epoch_samples
            train_acc = epoch_correct_sum / epoch_samples

            # Evaluation on test set
            outputs_test, _, Z_test = model.forward(X_test_T)
            test_loss = cross_entropy_batch(Y_test_T, Z_test[-1])
            test_acc = accuracy(Y_test_T, outputs_test)

            # Confusion matrix on test set 
            cm, y_true_labels, y_pred_labels = confusion_matrix(
            Y_test_T, outputs_test, num_classes=output_size
            )

            print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}"
            )

            # Log to WandB
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "learning_rate": lr,
                "batch_size": batch_size,
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_true_labels,
                    preds=y_pred_labels,
                    class_names=FASHION_MNIST_CLASSES
                )
            })

if __name__ == "__main__":
    main()
            




