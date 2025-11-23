import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer
from data_utils import fetch_papers, build_meta_path_adjs
from models.aggregator import SemanticAggregator, aggregate_meta_path
from models.fusion import SemanticFusion
from config import params
import random


# -------------------------------------------------------
# Generate positive and negative samples (BPR)
# -------------------------------------------------------
def generate_training_samples(num_items):
    samples = []
    all_items = list(range(num_items))

    for pos in all_items:
        neg = random.choice([i for i in all_items if i != pos])
        samples.append((pos, neg))

    return samples



# -------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------
def train_model(query):

    print(f"Training on query: {query}")

    # 1. Fetch papers
    works = fetch_papers(query, limit=40)
    if not works:
        print("No papers fetched.")
        return

    titles = [w.get("title", "") for w in works]

    # 2. Encode paper titles
    st = SentenceTransformer("intfloat/e5-base-v2")

    paper_emb = torch.tensor(
        st.encode(titles), dtype=torch.float, device=params["device"]
    )

    # 3. Build meta-path adjacencies
    pap_neighbors, pvp_neighbors = build_meta_path_adjs(works)

    # 4. Create model components
    aggregator_pap = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(params["device"])
    aggregator_pvp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(params["device"])

    proj_identity = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * params["L"]).to(params["device"])
    fusion = SemanticFusion(params["semantic_proj_dim"] * params["L"], params["semantic_proj_dim"]).to(params["device"])
    proj_user = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * params["L"]).to(params["device"])

    # 5. Optimizer
    model_params = (
        list(aggregator_pap.parameters()) +
        list(aggregator_pvp.parameters()) +
        list(proj_identity.parameters()) +
        list(fusion.parameters()) +
        list(proj_user.parameters())
    )
    optimizer = torch.optim.Adam(model_params, lr=1e-3)

    # 6. Training samples
    train_samples = generate_training_samples(len(works))

    # -------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------
    for epoch in range(10):
        total_loss = 0

        for pos, neg in train_samples:

            # ---- Forward pass is recomputed every iteration ----

            E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, aggregator_pap, L=params["L"])
            E_pvp = aggregate_meta_path(works, paper_emb, pvp_neighbors, aggregator_pvp, L=params["L"])
            E_identity = F.relu(proj_identity(paper_emb))

            fused_items, _ = fusion([E_identity, E_pap, E_pvp])
            fused_items = F.normalize(fused_items, dim=1)

            # Query encoding
            q_emb = torch.tensor(st.encode([query])[0], dtype=torch.float, device=params["device"])
            q_proj = F.normalize(proj_user(q_emb), dim=0)

            # ---- BPR Loss ----
            score_pos = torch.dot(fused_items[pos], q_proj)
            score_neg = torch.dot(fused_items[neg], q_proj)

            loss = -torch.log(torch.sigmoid(score_pos - score_neg))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} — Loss = {total_loss:.4f}")


    # Save model
    torch.save({
        "aggregator_pap": aggregator_pap.state_dict(),
        "aggregator_pvp": aggregator_pvp.state_dict(),
        "proj_identity": proj_identity.state_dict(),
        "fusion": fusion.state_dict(),
        "proj_user": proj_user.state_dict()
    }, "model_trained.pt")

    print("\nTraining completed successfully. Saved as model_trained.pt.\n")
