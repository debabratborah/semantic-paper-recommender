import torch

params = {
    "hidden_channels": 64,
    "st_dim": 768,
    "semantic_proj_dim": 128,
    "L": 2,
    "alpha": 0.6,
    "beta": 0.25,
    "gamma": 0.15,
    "epochs": 20,
    "lr": 5e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
