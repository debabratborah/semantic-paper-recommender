import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
import requests
import json
import os
from collections import defaultdict

# -------------------------
# Parameters
# -------------------------
params = {
    "hidden_channels": 64,        # used for any learnable projection dims (if used)
    "st_dim": 768,               # sentence-transformer dim (intfloat/e5-base-v2 -> 768)
    "semantic_proj_dim": 128,    # dim for attention projection (WU in paper)
    "L": 2,                      # number of stacked aggregation layers (high-order)
    "alpha": 0.6,                # fallback fusion weights (only used if attention not trained)
    "beta": 0.25,
    "gamma": 0.15,
    "epochs": 20,
    "lr": 5e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# -------------------------
# Utilities: fetch papers (richer fields)
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

    # cache
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
    else:
        cached = {}
    cached[query] = works
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f, indent=2)
    return works

# -------------------------
# Build simple meta-path adjacency helpers (P-A-P and P-V-P)
# -------------------------
def build_meta_path_adjs(works):
    """
    Returns:
      pap_neighbors: dict paper_idx -> set(paper_idx) via P-A-P
      pvp_neighbors: dict paper_idx -> set(paper_idx) via P-V-P
    """
    # map authors/venues to list of paper indices
    author_to_papers = defaultdict(list)
    venue_to_papers = defaultdict(list)
    for i, w in enumerate(works):
        for a in w.get("authors", []):
            if a.get("name"):
                author_to_papers[a["name"]].append(i)
        v = w.get("venue")
        if v:
            venue_to_papers[v].append(i)

    pap_neighbors = defaultdict(set)
    pvp_neighbors = defaultdict(set)

    # P-A-P: papers that share at least one author (via author)
    for author, paper_list in author_to_papers.items():
        for p in paper_list:
            pap_neighbors[p].update(paper_list)  # includes itself; we will handle self optionally

    # P-V-P: papers that share the same venue
    for venue, paper_list in venue_to_papers.items():
        for p in paper_list:
            pvp_neighbors[p].update(paper_list)

    # remove self from neighbor sets (optional)
    for p in range(len(works)):
        pap_neighbors[p].discard(p)
        pvp_neighbors[p].discard(p)

    return pap_neighbors, pvp_neighbors

# -------------------------
# Semantic Aggregation Layer (per meta-path)
# Implements Eq (2) (first-order) and stacking for high-order (Eq (3)-(4))
# e_phi^{1}_u = W1 eu + sum_{k in N_phi(u)} ( W1 ek + W2 (ek ⊙ eu) )
# We'll implement vectorized version using torch operations when possible.
# -------------------------
class SemanticAggregator(nn.Module):
    def __init__(self, st_dim, proj_dim):
        super().__init__()
        # projection matrices W_phi_1 and W_phi_2 shared across hops for simplicity
        self.W1 = nn.Linear(st_dim, proj_dim, bias=False)   # W_phi_1
        self.W2 = nn.Linear(st_dim, proj_dim, bias=False)   # W_phi_2
        self.bias = nn.Parameter(torch.zeros(proj_dim))     # b_phi (shared)

    def forward_first_order(self, center_embs, neighbor_embs_list):
        """
        center_embs: tensor (n_items, st_dim)
        neighbor_embs_list: list of neighbor embeddings *for each item*:
            - implemented as list[list of neighbor indices], but we'll vectorize externally.
        We'll implement a helper outside to gather neighbor aggregations.
        """
        raise NotImplementedError("Use aggregate_meta_path function below that coordinates neighbors.")

    def proj_W1(self, x):
        return self.W1(x)

    def proj_W2(self, x):
        return self.W2(x)

# -------------------------
# Semantic Fusion Layer (attention over meta-paths)
# Implements Eq (6)-(8)
# -------------------------
class SemanticFusion(nn.Module):
    def __init__(self, input_dim, attn_dim):
        super().__init__()
        # WU and bias bU from paper: project each semantic-specific embedding
        self.W = nn.Linear(input_dim, attn_dim, bias=True)   # WU and bU
        # attention vector qU (attn_dim)
        self.q = nn.Parameter(torch.randn(attn_dim))         # qU
        # dropout optional for stability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, semantic_embs):
        """
        semantic_embs: list of tensors [E_phi1 (n,d), E_phi2 (n,d), ...]  (same n and d)
        Returns: fused embeddings (n,d) after attention-weighted sum as Eq (8).
        """
        if not semantic_embs:
            return None
        # stack: (K, n, d) -> (n, K, d)
        stacked = torch.stack(semantic_embs, dim=1)  # (n, K, d)
        n, K, d = stacked.shape

        # project each embedding into attention space and compute score per node & per meta-path
        # reshape to (n*K, d) -> apply W -> tanh -> dot with q -> reshape to (n,K)
        proj = self.W(stacked.view(n * K, d))  # (n*K, attn_dim)
        proj = torch.tanh(proj)                # activation
        proj = proj.view(n, K, -1)             # (n, K, attn_dim)

        # dot with q: (n,K,attn_dim) dot (attn_dim,) -> (n,K)
        scores = torch.matmul(proj, self.q)    # (n,K)
        # softmax across meta-paths (K dim)
        weights = F.softmax(scores, dim=1)     # (n,K)

        # weighted sum: fused[i] = sum_k weights[i,k] * stacked[i,k]
        weights_exp = weights.unsqueeze(-1)    # (n,K,1)
        fused = (weights_exp * stacked).sum(dim=1)  # (n,d)
        fused = self.dropout(fused)
        return fused, weights  # return weights for interpretability

# -------------------------
# Helper: aggregate multi-hop meta-path embeddings
# We'll implement stacked aggregation as described:
#   e^{1} = A1(center, neighbors)
#   e^{2} = A2(e^{1}, neighbors_of_e1)  etc.
# For simplicity (and given small toy graph), we'll apply the paper's first-order formula
# iteratively L times, each time treating the previous order embedding as 'center'.
# -------------------------
def aggregate_meta_path(works, st_embeddings, neighbors_dict, aggregator: SemanticAggregator, L=2):
    """
    works: list of works
    st_embeddings: torch.Tensor (n_items, st_dim) sentence-transformer embeddings
    neighbors_dict: dict: paper_idx -> set(paper_idx) (meta-path neighbors)
    aggregator: SemanticAggregator instance
    L: number of stacked aggregations
    Returns: semantic_specific_embeddings: tensor (n_items, d_concat)
             where we concatenate e^{1} || e^{2} || ... || e^{L} (Eq (4))
    """
    n = len(works)
    device = st_embeddings.device
    st_dim = st_embeddings.size(1)
    proj_dim = aggregator.W1.out_features

    # Prepare neighbor index lists for quick gather
    # For each paper, create list of neighbor indices (may be empty)
    neighbor_idx_list = [torch.tensor(sorted(list(neighbors_dict.get(i, set()))), dtype=torch.long, device=device)
                         for i in range(n)]

    # Initialize center embeddings for layer 0 as original st embeddings
    center = st_embeddings  # (n, st_dim)

    order_embeddings = []
    for layer in range(1, L + 1):
        # For each item i, compute:
        # e_phi,layer[i] = W1 center[i] + sum_{k in N(i)} ( W1 ek + W2 (ek ⊙ center[i]) )
        # We'll compute vectorized where possible by looping items (graph is small)
        out_layer = torch.zeros((n, proj_dim), device=device)
        W1 = aggregator.W1
        W2 = aggregator.W2

        # precompute W1(center) (n,proj_dim)
        W1_center = W1(center)  # (n, proj_dim)

        for i in range(n):
            ci = center[i]            # (st_dim,)
            base = W1_center[i]      # (proj_dim,)
            neigh_idx = neighbor_idx_list[i]
            if neigh_idx.numel() == 0:
                agg = torch.zeros_like(base)
            else:
                neigh_vecs = st_embeddings[neigh_idx]  # neighbor st embeddings (m, st_dim)
                # W1 ek
                w1_neigh = W1(neigh_vecs)              # (m, proj_dim)
                # W2 (ek ⊙ center[i])
                # elementwise product: (m, st_dim) * (st_dim,) -> (m, st_dim)
                ek_x_eu = neigh_vecs * ci.unsqueeze(0)
                w2_part = W2(ek_x_eu)                  # (m, proj_dim)
                # sum over neighbors
                agg = (w1_neigh + w2_part).sum(dim=0)  # (proj_dim,)
            out_layer[i] = base + agg + aggregator.bias
        # activation (paper uses linear then concatenation; we'll use ReLU between layers)
        out_layer = F.relu(out_layer)
        order_embeddings.append(out_layer)   # each is (n, proj_dim)
        # for next layer, set center to a mapping from out_layer back to st_dim space
        # but to keep things simple and stable, we set center to a reconstructed vector:
        # project back via a linear layer: here we simply map with a pseudo-inverse-like linear map
        # For simplicity, let center = tanh( linear_proj(out_layer) ) where linear_proj maps proj_dim -> st_dim
        # We'll create a temporary linear mapping per call (not learned across calls) using random projection for now.
        # To keep implementable without extra params, set center = tanh( out_layer @ W_back ), where W_back is random fixed.
        # (Note: In a fully trainable model, W_back would be learned.)
        W_back = torch.randn((proj_dim, st_dim), device=device) * (1.0 / (proj_dim ** 0.5))
        center = torch.tanh(out_layer @ W_back)  # (n, st_dim)

    # Concatenate e^{1} || e^{2} || ... || e^{L} along dim=1 => (n, proj_dim * L)
    semantic_specific = torch.cat(order_embeddings, dim=1) if order_embeddings else torch.zeros((n, proj_dim * L), device=device)
    return semantic_specific  # (n, proj_dim * L)

# -------------------------
# Main flow: fetch, encode nodes, build meta-path adjs, aggregate, fuse, predict
# -------------------------
def main():
    query = input("Enter your search query: ").strip()
    top_k = 10
    print(f"--- Using device: {params['device']} ---")

    # 1) fetch works
    works = fetch_papers(query, limit=30)

    if len(works) == 0:
        print("No works fetched for the query.")
        return

    # 2) prepare textual lists
    titles = [w.get("title", "") or "" for w in works]
    authors = list({a.get("name") for w in works for a in w.get("authors", []) if a.get("name")})
    venues = list({w.get("venue") for w in works if w.get("venue")})

    author_to_idx = {name: i for i, name in enumerate(authors)}
    venue_to_idx = {v: i for i, v in enumerate(venues)}

    # 3) encode textual nodes using sentence-transformer (semantic initialization in paper was random lookup,
    #    but we keep ST embeddings so we retain your earlier semantic cues while applying HGRec aggregation)
    print("[DEBUG] Loading SentenceTransformer and encoding nodes...")
    st_model = SentenceTransformer("intfloat/e5-base-v2")
    paper_emb = torch.tensor(st_model.encode(titles, convert_to_tensor=False), dtype=torch.float, device=params["device"])
    author_emb = torch.tensor(st_model.encode(authors, convert_to_tensor=False), dtype=torch.float, device=params["device"]) if authors else torch.empty((0, params["st_dim"]), device=params["device"])
    venue_emb = torch.tensor(st_model.encode(venues, convert_to_tensor=False), dtype=torch.float, device=params["device"]) if venues else torch.empty((0, params["st_dim"]), device=params["device"])

    n_papers = len(works)
    print(f"[DEBUG] {n_papers} papers, {len(authors)} authors, {len(venues)} venues encoded.")

    # 4) build meta-path adjacency P-A-P and P-V-P
    pap_neighbors, pvp_neighbors = build_meta_path_adjs(works)

    # 5) instantiate aggregation modules for each meta-path
    aggregator_pap = SemanticAggregator(st_dim=params["st_dim"], proj_dim=params["semantic_proj_dim"]).to(params["device"])
    aggregator_pvp = SemanticAggregator(st_dim=params["st_dim"], proj_dim=params["semantic_proj_dim"]).to(params["device"])

    # 6) aggregate multi-hop embeddings per meta-path
    L = params["L"]
    print(f"[DEBUG] Aggregating meta-path embeddings with L={L} ...")
    E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, aggregator_pap, L=L)  # (n, proj_dim * L)
    E_pvp = aggregate_meta_path(works, paper_emb, pvp_neighbors, aggregator_pvp, L=L)  # (n, proj_dim * L)

    # For interpretability and stability, we can also add original paper projection as another 'meta-path' (identity)
    # Project original paper embedding into same dim (proj_dim * L) by applying linear layers (here simple repeat/proj)
    # Create identity-like semantic specific embedding
    # We'll tile/linear-project paper_emb into (n, proj_dim * L)
    proj_identity = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * L).to(params["device"])
    E_identity = F.relu(proj_identity(paper_emb))  # (n, proj_dim * L)

    # 7) semantic fusion over meta-paths
    # Stack meta-path-specific embeddings: [E_identity, E_pap, E_pvp]
    semantic_embs = [E_identity, E_pap, E_pvp]
    fusion = SemanticFusion(input_dim=E_identity.size(1), attn_dim=params["semantic_proj_dim"]).to(params["device"])
    fused_items, meta_weights = fusion(semantic_embs)  # fused_items: (n, d) where d = proj_dim * L

    # normalize fused item embeddings (so inner product behaves like cosine)
    fused_items = F.normalize(fused_items, dim=1)

    # 8) create query embedding (treat query as user embedding)
    q_emb = torch.tensor(st_model.encode([query], convert_to_tensor=False)[0], dtype=torch.float, device=params["device"])
    q_emb = F.normalize(q_emb, dim=0)  # normalized user embedding

    # 9) Project query embedding into same fused item dimension for inner product:
    # We'll use a small projection M to map st_dim -> fused_dim (learnable). If not trained, use simple linear.
    proj_user = nn.Linear(params["st_dim"], fused_items.size(1)).to(params["device"])
    user_proj = proj_user(q_emb)  # (fused_dim,)
    user_proj = F.normalize(user_proj, dim=0)

    # 10) final prediction: inner product y_hat_ui = E_u^T E_i
    scores = torch.mv(fused_items, user_proj)  # (n_items,)
    topk = torch.topk(scores, k=min(top_k, scores.size(0)))
    print("\n--- Top recommendations (HGRec-style fused) ---")
    for rank, (idx, sc) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
        paper = works[idx]
        title = paper.get("title", "N/A")
        authors_print = [a.get("name") for a in paper.get("authors", [])][:4]
        print(f"{rank}. {title} (score: {sc:.4f})")
        print(f"   Authors: {authors_print}")
        print(f"   Venue: {paper.get('venue','N/A')}, Year: {paper.get('year','N/A')}")
        print(f"   URL: {paper.get('url','N/A')}\n")

    # 11) show per-item meta-path weights for first top recommendation (interpretability)
    # meta_weights: (n, K) where K=3 here
    print("[DEBUG] Meta-path attention weights for top recommended paper:")
    w_top = meta_weights[topk.indices[0]] if meta_weights is not None else None
    if w_top is not None:
        print(f" weights (identity, P-A-P, P-V-P): {w_top.detach().cpu().numpy()}")

    # ---- Note: BPR training (requires user-item interactions) ----
    # The paper optimizes the model using BPR loss (Eq. 10). To train this end-to-end:
    #  - you need a dataset of observed interactions (u, i) and negative samples (u, j)
    #  - sample triplets (u, i, j), compute Eu and Ei (using same fusion pipeline), compute y_ui, y_uj
    #  - compute loss = -ln sigma(y_ui - y_uj) + lambda * ||Theta||^2
    #  - optimize parameters (aggregators' W1/W2, fusion W/q, projection layers, etc.)
    #
    # I can add a full BPR training loop if you provide/choose an interaction dataset (e.g., Movielens).
    # -----------------------------------------------------------------

if __name__ == "__main__":
    main()
