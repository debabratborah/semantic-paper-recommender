import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer
import requests
import json
import os
from collections import defaultdict
import numpy as np

# -------------------------
# Parameters
# -------------------------
params = {
    "st_dim": 768,
    "semantic_proj_dim": 128,
    "L": 2,
    "top_k": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# -------------------------
# Utilities
# -------------------------
def fetch_papers(query, limit=25, cache_file="cached_works.json"):
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if query in cached:
            return cached[query]

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

class SemanticAggregator(nn.Module):
    def __init__(self, st_dim, proj_dim):
        super().__init__()
        self.W1 = nn.Linear(st_dim, proj_dim, bias=False)
        self.W2 = nn.Linear(st_dim, proj_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(proj_dim))

class SemanticFusion(nn.Module):
    def __init__(self, input_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, attn_dim, bias=True)
        self.q = nn.Parameter(torch.randn(attn_dim))
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, semantic_embs):
        if not semantic_embs:
            return None
        stacked = torch.stack(semantic_embs, dim=1)
        n, K, d = stacked.shape
        proj = torch.tanh(self.W(stacked.view(n*K, d)))
        proj = proj.view(n, K, -1)
        scores = torch.matmul(proj, self.q)
        weights = F.softmax(scores, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        fused = self.dropout(fused)
        return fused, weights

def aggregate_meta_path(works, st_embeddings, neighbors_dict, aggregator: SemanticAggregator, L=2):
    n = len(works)
    device = st_embeddings.device
    st_dim = st_embeddings.size(1)
    proj_dim = aggregator.W1.out_features
    neighbor_idx_list = [torch.tensor(sorted(list(neighbors_dict.get(i, set()))), dtype=torch.long, device=device)
                         for i in range(n)]
    center = st_embeddings
    order_embeddings = []
    for layer in range(1, L+1):
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
        W_back = torch.randn((proj_dim, st_dim), device=device) * (1.0/(proj_dim**0.5))
        center = torch.tanh(out_layer @ W_back)
    return torch.cat(order_embeddings, dim=1) if order_embeddings else torch.zeros((n, proj_dim*L), device=device)

# -------------------------
# Metrics
# -------------------------
def precision_at_k(ranked_idx, relevant_idx, k):
    return len(set(ranked_idx[:k]) & set(relevant_idx)) / k

def recall_at_k(ranked_idx, relevant_idx, k):
    return len(set(ranked_idx[:k]) & set(relevant_idx)) / len(relevant_idx) if relevant_idx else 0.0

def hr_at_k(ranked_idx, relevant_idx, k):
    return int(len(set(ranked_idx[:k]) & set(relevant_idx)) > 0)

def ndcg_at_k(ranked_idx, relevant_idx, k):
    dcg = 0.0
    for i, idx in enumerate(ranked_idx[:k]):
        if idx in relevant_idx:
            dcg += 1.0 / np.log2(i+2)
    idcg = sum(1.0/np.log2(i+2) for i in range(min(len(relevant_idx), k)))
    return dcg / idcg if idcg > 0 else 0.0

# -------------------------
# Recommendation function
# -------------------------
def recommend(works, paper_emb, pap_neighbors, use_fusion=True):
    device = params["device"]
    aggregator_pap = SemanticAggregator(st_dim=params["st_dim"], proj_dim=params["semantic_proj_dim"]).to(device)
    E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, aggregator_pap, L=params["L"])

    if use_fusion:
        proj_identity = nn.Linear(params["st_dim"], params["semantic_proj_dim"]*params["L"]).to(device)
        E_identity = F.relu(proj_identity(paper_emb))
        semantic_embs = [E_identity, E_pap]
        fusion = SemanticFusion(input_dim=E_identity.size(1), attn_dim=params["semantic_proj_dim"]).to(device)
        fused_items, _ = fusion(semantic_embs)
    else:
        fused_items = E_pap

    fused_items = F.normalize(fused_items, dim=1)
    return fused_items

# -------------------------
# Batch evaluation
# -------------------------
def batch_evaluation(queries):
    st_model = SentenceTransformer("intfloat/e5-base-v2")
    results_hgrec = {"Precision":[], "Recall":[], "NDCG":[], "HR":[]}
    results_minus = {"Precision":[], "Recall":[], "NDCG":[], "HR":[]}

    for query in queries:
        works = fetch_papers(query, limit=50)
        if len(works) == 0:
            continue
        titles = [w.get("title","") for w in works]
        paper_emb = torch.tensor(st_model.encode(titles, convert_to_tensor=False), dtype=torch.float, device=params["device"])
        pap_neighbors = build_pap_adjs(works)
        test_interactions = list(range(min(5,len(works))))

        # HGRec
        fused_hgrec = recommend(works, paper_emb, pap_neighbors, use_fusion=True)
        q_emb = torch.tensor(st_model.encode([query], convert_to_tensor=False)[0], dtype=torch.float, device=params["device"])
        q_emb = F.normalize(q_emb, dim=0)
        proj_user = nn.Linear(params["st_dim"], fused_hgrec.size(1)).to(params["device"])
        user_proj = F.normalize(proj_user(q_emb), dim=0)
        scores = torch.mv(fused_hgrec, user_proj)
        ranked_idx = torch.argsort(scores, descending=True).tolist()

        results_hgrec["Precision"].append(precision_at_k(ranked_idx,test_interactions,params["top_k"]))
        results_hgrec["Recall"].append(recall_at_k(ranked_idx,test_interactions,params["top_k"]))
        results_hgrec["NDCG"].append(ndcg_at_k(ranked_idx,test_interactions,params["top_k"]))
        results_hgrec["HR"].append(hr_at_k(ranked_idx,test_interactions,params["top_k"]))

        # HGRec-minus
        fused_minus = recommend(works, paper_emb, pap_neighbors, use_fusion=False)
        scores_minus = torch.mv(fused_minus, user_proj)
        ranked_idx_minus = torch.argsort(scores_minus, descending=True).tolist()

        results_minus["Precision"].append(precision_at_k(ranked_idx_minus,test_interactions,params["top_k"]))
        results_minus["Recall"].append(recall_at_k(ranked_idx_minus,test_interactions,params["top_k"]))
        results_minus["NDCG"].append(ndcg_at_k(ranked_idx_minus,test_interactions,params["top_k"]))
        results_minus["HR"].append(hr_at_k(ranked_idx_minus,test_interactions,params["top_k"]))

    # Compute average metrics
    print("\n--- Average Metrics over all queries ---")
    print("HGRec (full):")
    for k,v in results_hgrec.items():
        print(f"{k}@{params['top_k']}: {np.mean(v):.4f}")
    print("\nHGRec-minus (P-A-P only):")
    for k,v in results_minus.items():
        print(f"{k}@{params['top_k']}: {np.mean(v):.4f}")

if __name__ == "__main__":
    queries = [
        "Graph Neural Networks",
        "Heterogeneous Graph Embedding",
        "Recommender Systems",
        "Deep Learning for Graphs"
    ]
    batch_evaluation(queries)
