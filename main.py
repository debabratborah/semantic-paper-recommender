from config import params
from data_utils import fetch_papers, build_meta_path_adjs
from models.aggregator import SemanticAggregator, aggregate_meta_path
from models.fusion import SemanticFusion
import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer

def main():
    query = input("Enter your search query: ").strip()
    print(f"--- Using device: {params['device']} ---")
    works = fetch_papers(query, limit=30)

    if not works:
        print("No works fetched for the query.")
        return

    titles = [w.get("title", "") or "" for w in works]
    authors = list({a.get("name") for w in works for a in w.get("authors", []) if a.get("name")})
    venues = list({w.get("venue") for w in works if w.get("venue")})

    print("[DEBUG] Loading SentenceTransformer and encoding nodes...")
    st_model = SentenceTransformer("intfloat/e5-base-v2")
    paper_emb = torch.tensor(st_model.encode(titles), dtype=torch.float, device=params["device"])
    author_emb = torch.tensor(st_model.encode(authors), dtype=torch.float, device=params["device"]) if authors else torch.empty((0, params["st_dim"]), device=params["device"])
    venue_emb = torch.tensor(st_model.encode(venues), dtype=torch.float, device=params["device"]) if venues else torch.empty((0, params["st_dim"]), device=params["device"])

    print(f"[DEBUG] {len(works)} papers, {len(authors)} authors, {len(venues)} venues encoded.")

    pap_neighbors, pvp_neighbors = build_meta_path_adjs(works)
    aggregator_pap = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(params["device"])
    aggregator_pvp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(params["device"])

    E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, aggregator_pap, L=params["L"])
    E_pvp = aggregate_meta_path(works, paper_emb, pvp_neighbors, aggregator_pvp, L=params["L"])

    proj_identity = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * params["L"]).to(params["device"])
    E_identity = F.relu(proj_identity(paper_emb))
    fusion = SemanticFusion(E_identity.size(1), params["semantic_proj_dim"]).to(params["device"])
    fused_items, meta_weights = fusion([E_identity, E_pap, E_pvp])
    fused_items = F.normalize(fused_items, dim=1)

    q_emb = torch.tensor(st_model.encode([query])[0], dtype=torch.float, device=params["device"])
    q_emb = F.normalize(q_emb, dim=0)
    proj_user = nn.Linear(params["st_dim"], fused_items.size(1)).to(params["device"])
    user_proj = F.normalize(proj_user(q_emb), dim=0)

    scores = torch.mv(fused_items, user_proj)
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
    main()
