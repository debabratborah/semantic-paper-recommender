import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer
import requests
import json
import os
from collections import defaultdict

# -------------------------
# Parameters
# -------------------------
params = {
    "st_dim": 768,               # sentence-transformer dimension
    "semantic_proj_dim": 128,    # projection dim for aggregation
    "L": 2,                      # stacked aggregation layers
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# -------------------------
# Utilities: fetch papers
# -------------------------
def fetch_papers(query, limit=25, cache_file="cached_works.json"):
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if query in cached:
            print(f"[DEBUG] Using cached results for query: '{query}'")
            return cached[query]

    print(f"[DEBUG] Fetching papers from Semantic Scholar for query: '{query}'")
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,abstract,authors,venue,url,year"
    params_req = {"query": query, "limit": limit, "fields": fields}
    resp = requests.get(base, params=params_req)
    resp.raise_for_status()
    works = resp.json().get("data", [])

    cached = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
    cached[query] = works
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f, indent=2)
    return works

# -------------------------
# Build P-A-P adjacency
# -------------------------
def build_pap_adjs(works):
    author_to_papers = defaultdict(list)
    for i, w in enumerate(works):
        for a in w.get("authors", []):
            if a.get("name"):
                author_to_papers[a["name"]].append(i)

    pap_neighbors = defaultdict(set)
    for paper_list in author_to_papers.values():
        for p in paper_list:
            pap_neighbors[p].update(paper_list)

    for p in range(len(works)):
        pap_neighbors[p].discard(p)
    return pap_neighbors

# -------------------------
# Semantic Aggregation Layer
# -------------------------
class SemanticAggregator(nn.Module):
    def __init__(self, st_dim, proj_dim):
        super().__init__()
        self.W1 = nn.Linear(st_dim, proj_dim, bias=False)
        self.W2 = nn.Linear(st_dim, proj_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(proj_dim))

# -------------------------
# Multi-hop aggregation
# -------------------------
def aggregate_meta_path(works, st_embeddings, neighbors_dict, aggregator: SemanticAggregator, L=2):
    n = len(works)
    device = st_embeddings.device
    st_dim = st_embeddings.size(1)
    proj_dim = aggregator.W1.out_features

    neighbor_idx_list = [torch.tensor(sorted(list(neighbors_dict.get(i, set()))), dtype=torch.long, device=device)
                         for i in range(n)]
    center = st_embeddings
    order_embeddings = []

    for layer in range(1, L + 1):
        out_layer = torch.zeros((n, proj_dim), device=device)
        W1 = aggregator.W1
        W2 = aggregator.W2
        W1_center = W1(center)

        for i in range(n):
            ci = center[i]
            base = W1_center[i]
            neigh_idx = neighbor_idx_list[i]
            if neigh_idx.numel() == 0:
                agg = torch.zeros_like(base)
            else:
                neigh_vecs = st_embeddings[neigh_idx]
                w1_neigh = W1(neigh_vecs)
                ek_x_eu = neigh_vecs * ci.unsqueeze(0)
                w2_part = W2(ek_x_eu)
                agg = (w1_neigh + w2_part).sum(dim=0)
            out_layer[i] = base + agg + aggregator.bias
        out_layer = F.relu(out_layer)
        order_embeddings.append(out_layer)

        W_back = torch.randn((proj_dim, st_dim), device=device) * (1.0 / (proj_dim ** 0.5))
        center = torch.tanh(out_layer @ W_back)

    semantic_specific = torch.cat(order_embeddings, dim=1) if order_embeddings else torch.zeros((n, proj_dim * L), device=device)
    return semantic_specific

# -------------------------
# Main HGRec-minus flow
# -------------------------
def main_minus():
    query = input("Enter your search query: ").strip()
    top_k = 10
    print(f"--- Using device: {params['device']} ---")

    works = fetch_papers(query, limit=30)
    if len(works) == 0:
        print("No works fetched.")
        return

    titles = [w.get("title", "") or "" for w in works]

    print("[DEBUG] Loading SentenceTransformer...")
    st_model = SentenceTransformer("intfloat/e5-base-v2")
    paper_emb = torch.tensor(st_model.encode(titles, convert_to_tensor=False), dtype=torch.float, device=params["device"])

    n_papers = len(works)
    print(f"[DEBUG] {n_papers} papers encoded.")

    pap_neighbors = build_pap_adjs(works)
    aggregator_pap = SemanticAggregator(st_dim=params["st_dim"], proj_dim=params["semantic_proj_dim"]).to(params["device"])

    print(f"[DEBUG] Aggregating P-A-P embeddings with L={params['L']} ...")
    E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, aggregator_pap, L=params["L"])
    fused_items = F.normalize(E_pap, dim=1)

    # Query embedding
    q_emb = torch.tensor(st_model.encode([query], convert_to_tensor=False)[0], dtype=torch.float, device=params["device"])
    q_emb = F.normalize(q_emb, dim=0)

    proj_user = nn.Linear(params["st_dim"], fused_items.size(1)).to(params["device"])
    user_proj = proj_user(q_emb)
    user_proj = F.normalize(user_proj, dim=0)

    # Scores
    scores = torch.mv(fused_items, user_proj)
    topk = torch.topk(scores, k=min(top_k, scores.size(0)))

    print("\n--- Top recommendations (HGRec-minus: P-A-P only) ---")
    for rank, (idx, sc) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
        paper = works[idx]
        title = paper.get("title", "N/A")
        authors_print = [a.get("name") for a in paper.get("authors", [])][:4]
        print(f"{rank}. {title} (score: {sc:.4f})")
        print(f"   Authors: {authors_print}")
        print(f"   Venue: {paper.get('venue','N/A')}, Year: {paper.get('year','N/A')}")
        print(f"   URL: {paper.get('url','N/A')}\n")

if __name__ == "__main__":
    main_minus()
