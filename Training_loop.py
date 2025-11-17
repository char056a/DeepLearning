import numpy as np 

from FFNN import FFNN 
from Actfunc import Relu
from init import xavier_uniform
from Data_loading import FASHION_MNIST
from Loss import cross_entropy_batch

import wandb

# Compute accuracy for a batch
def accuracy(Y_true, logits):
    # logits = network output before softmax

    preds = np.argmax(logits, axis=0)
    targets = np.argmax(Y_true, axis=0)
    return np.mean(preds == targets)

# Entry point for training the model
def main():
    # Log in to WandB
    wandb.login()

    # Define hyperparameters that we want to track
    config = {
        "epochs": 10,
        "learning_rate": 0.01,
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

        # Training loop
        for epoch in range(cfg.epochs):
            # Forward pass on full training set
            outputs_train, A_train, Z_train = model.forward(X_train_T)

            # Compute training loss
            train_loss = cross_entropy_batch(Y_train_T, Z_train[-1])

            # Compute accuracy
            train_acc = accuracy(Y_train_T, outputs_train)

            # Backprop: Compute gradients
            grads_w, grads_b = model.full_gradient(
                A_train,    # activations of each layer
                Z_train,    # logits of each layer
                Y_train_T,  # one-hot labels
                X_train_T   # input data
            )

            # Gradient descent update 
            for layer, dW, dB in zip(model.layers, grads_w, grads_b):
                layer.weights -= lr * dW
                layer.bias -= lr * dB

            # Evaluate on test set
            outputs_test, _, Z_test = model.forward(X_test_T)
            test_loss = cross_entropy_batch(Y_test_T, Z_test[-1])
            test_acc = accuracy(Y_test_T, outputs_test)

            # Log metrics to WandB
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "learning_rate": lr,
            })

if __name__ == "__main__":
    main()
            




