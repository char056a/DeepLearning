import wandb

wandb.login()

# Project that the run is recorded to
project = "..."

# Define hyperparameters 
config = {
    "epochs": 10,
    "learning_rate": 0.01,
    "hidden_sizes": [2,2],
    "activation": "Relu",
    "init": "xavier_uniform"
}

# Initialize run 

