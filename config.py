import torch

params = {

    # =========================================================
    # Dataset
    # =========================================================

    # using OpenAlex dataset collected through API
    "dataset": "openalex",

    # number of papers used from papers.json
    "subset_size": 15000,


    # =========================================================
    # Embedding dimensions
    # =========================================================

    # sentence transformer embedding size
    # (e5-base-v2 or similar)
    "st_dim": 384,

    # projection dimension used inside SemanticAggregator
    "semantic_proj_dim": 128,

    # semantic attention dimension for fusion layer
    "attn_dim": 64,


    # =========================================================
    # Graph aggregation
    # =========================================================

    # number of HGNN propagation layers
    "L": 2,


    # =========================================================
    # Meta-path weights (optional initialization)
    # =========================================================

    # Paper → Citation → Paper
    "alpha": 0.35,

    # Paper → Author → Paper
    "beta": 0.25,

    # Paper → Keyword / Concept → Paper
    "gamma": 0.20,

    # Paper → Venue → Paper
    "delta": 0.10,

    # Paper → Year → Paper
    "epsilon": 0.10,


    # =========================================================
    # Training parameters
    # =========================================================

    "epochs": 25,

    "lr": 3e-4,

    "weight_decay": 1e-5,


    # =========================================================
    # Recommendation parameters
    # =========================================================

    # number of recommended papers returned
    "top_k": 10,


    # =========================================================
    # Device
    # =========================================================

    "device": "cpu",
}