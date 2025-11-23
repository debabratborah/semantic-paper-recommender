from config import params
from data_utils import fetch_papers, build_meta_path_adjs
from models.aggregator import SemanticAggregator, aggregate_meta_path
from models.fusion import SemanticFusion
from train import train_model
from visualize_graph import visualize_graph         # <--- ADD THIS
import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer


def run_inference(query):

    print(f"--- Using device: {params['device']} ---")
    works = fetch_papers(query, limit=30)

    if not works:
        print("No works fetched for the query.")
        return

    titles = [w.get("title", "") for w in works]
    st_model = SentenceTransformer("intfloat/e5-base-v2")

    paper_emb = torch.tensor(
        st_model.encode(titles), dtype=torch.float, device=params["device"]
    )

    # ==============================
    # Build meta-path adjacency
    # ==============================
    pap_neighbors, pvp_neighbors = build_meta_path_adjs(works)

    # --- VISUALIZE GRAPH HERE ---
    visualize_graph(works, pap_neighbors, pvp_neighbors)
    # ==============================
    # Continue with inference
    # ==============================

    aggregator_pap = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(params["device"])
    aggregator_pvp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(params["device"])
    proj_identity = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * params["L"]).to(params["device"])
    fusion = SemanticFusion(params["semantic_proj_dim"] * params["L"], params["semantic_proj_dim"]).to(params["device"])
    proj_user = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * params["L"]).to(params["device"])

    try:
        ckpt = torch.load("model_trained.pt", map_location=params["device"])
        aggregator_pap.load_state_dict(ckpt["aggregator_pap"])
        aggregator_pvp.load_state_dict(ckpt["aggregator_pvp"])
        proj_identity.load_state_dict(ckpt["proj_identity"])
        fusion.load_state_dict(ckpt["fusion"])
        proj_user.load_state_dict(ckpt["proj_user"])
        print("[DEBUG] Loaded trained model weights.")
    except:
        print("[WARNING] No trained model found — using random weights.")


    E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, aggregator_pap, L=params["L"])
    E_pvp = aggregate_meta_path(works, paper_emb, pvp_neighbors, aggregator_pvp, L=params["L"])
    E_identity = F.relu(proj_identity(paper_emb))

    fused_items, meta_weights = fusion([E_identity, E_pap, E_pvp])
    fused_items = F.normalize(fused_items, dim=1)

    q_emb = torch.tensor(st_model.encode([query])[0], dtype=torch.float, device=params["device"])
    q_proj = F.normalize(proj_user(q_emb), dim=0)

    scores = torch.mv(fused_items, q_proj)
    topk = torch.topk(scores, k=min(10, scores.size(0)))

    print("\n--- Top Recommendations ---")
    for rank, (idx, sc) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
        paper = works[idx]
        print(f"{rank}. {paper.get('title','N/A')} (score: {sc:.4f})")
        print(f"   Authors: {[a.get('name') for a in paper.get('authors', [])][:4]}")
        print(f"   Venue: {paper.get('venue','N/A')}, Year: {paper.get('year','N/A')}")
        print(f"   URL: {paper.get('url','N/A')}\n")

    if meta_weights is not None:
        print("[DEBUG] Meta-path attention weights for top paper:")
        print(meta_weights[topk.indices[0]].detach().cpu().numpy())


if __name__ == "__main__":
    mode = input("Train or Inference? (t/i): ").strip().lower()

    if mode == "t":
        query = input("Enter training query: ")
        train_model(query)
    else:
        query = input("Enter your search query: ")
        run_inference(query)
