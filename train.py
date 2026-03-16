import torch
import torch.nn.functional as F
from torch import nn

from config import params
from data_utils import fetch_papers, build_meta_path_adjs

from models.aggregator import SemanticAggregator, aggregate_meta_path
from models.fusion import SemanticFusion


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    works = fetch_papers(query=None, limit=params["subset_size"])

    print("Loaded papers:", len(works))

    return works


# =========================================================
# LOAD PRECOMPUTED EMBEDDINGS
# =========================================================

def load_embeddings(device):

    print("Loading precomputed embeddings...")

    paper_emb = torch.load("paper_embeddings.pt")

    paper_emb = paper_emb.to(device)

    print("Embedding shape:", paper_emb.shape)

    return paper_emb


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model():

    device = params["device"]

    # ------------------------------------------
    # Load dataset
    # ------------------------------------------

    works = load_dataset()

    # ------------------------------------------
    # Load embeddings
    # ------------------------------------------

    paper_emb = load_embeddings(device)

    # ------------------------------------------
    # Build meta-path graphs
    # ------------------------------------------

    (
        pap_neighbors,
        pvp_neighbors,
        pyp_neighbors,
        pkp_neighbors,
        pcp_neighbors
    ) = build_meta_path_adjs(works)

    # ------------------------------------------
    # Initialize aggregators
    # ------------------------------------------

    ag_pap = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pvp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pyp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pkp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pcp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)

    # identity projection
    proj_identity = nn.Linear(
        params["st_dim"],
        params["semantic_proj_dim"] * params["L"]
    ).to(device)

    # semantic fusion
    fusion = SemanticFusion(
        params["semantic_proj_dim"] * params["L"],
        params["attn_dim"]
    ).to(device)

    # ------------------------------------------
    # Optimizer
    # ------------------------------------------

    optimizer = torch.optim.Adam(
        list(ag_pap.parameters()) +
        list(ag_pvp.parameters()) +
        list(ag_pyp.parameters()) +
        list(ag_pkp.parameters()) +
        list(ag_pcp.parameters()) +
        list(proj_identity.parameters()) +
        list(fusion.parameters()),
        lr=params["lr"]
    )

    # ------------------------------------------
    # Training loop
    # ------------------------------------------

    for epoch in range(params["epochs"]):

        optimizer.zero_grad()

        # meta-path embeddings

        E_pap = aggregate_meta_path(
            works, paper_emb, pap_neighbors, ag_pap, L=params["L"]
        )

        E_pvp = aggregate_meta_path(
            works, paper_emb, pvp_neighbors, ag_pvp, L=params["L"]
        )

        E_pyp = aggregate_meta_path(
            works, paper_emb, pyp_neighbors, ag_pyp, L=params["L"]
        )

        E_pkp = aggregate_meta_path(
            works, paper_emb, pkp_neighbors, ag_pkp, L=params["L"]
        )

        E_pcp = aggregate_meta_path(
            works, paper_emb, pcp_neighbors, ag_pcp, L=params["L"]
        )

        # identity embeddings
        E_identity = torch.relu(proj_identity(paper_emb))

        # semantic fusion
        fused_items, weights = fusion([
            E_identity,
            E_pap,
            E_pvp,
            E_pyp,
            E_pkp,
            E_pcp
        ])

        fused_items = F.normalize(fused_items, dim=1)

        # similarity matrix
        sim_matrix = torch.matmul(fused_items, fused_items.T)

        loss = -torch.mean(torch.log(torch.sigmoid(sim_matrix)))

        loss.backward()

        optimizer.step()

        print(f"Epoch {epoch+1}/{params['epochs']}  Loss: {loss.item():.4f}")

    # ------------------------------------------
    # Save model
    # ------------------------------------------

    torch.save({

        "aggregator_pap": ag_pap.state_dict(),
        "aggregator_pvp": ag_pvp.state_dict(),
        "aggregator_pyp": ag_pyp.state_dict(),
        "aggregator_pkp": ag_pkp.state_dict(),
        "aggregator_pcp": ag_pcp.state_dict(),

        "proj_identity": proj_identity.state_dict(),

        "fusion": fusion.state_dict()

    }, "model_trained.pt")

    print("Model saved → model_trained.pt")


# =========================================================
# RUN TRAINING
# =========================================================

if __name__ == "__main__":

    train_model()