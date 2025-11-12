import torch
import torch.nn.functional as F
from torch import nn

# ================================================================
#  CLASS: SemanticAggregator
#  PURPOSE:
#     - Learns to combine node and neighbor embeddings
#     - Uses two linear projections (W1, W2)
#     - Encodes both individual node features and pairwise interactions
# ================================================================
class SemanticAggregator(nn.Module):
    def __init__(self, st_dim, proj_dim):
        super().__init__()
        # W1 ∈ ℝ^(proj_dim × st_dim)
        # Projects node (and neighbor) embeddings to latent space
        self.W1 = nn.Linear(st_dim, proj_dim, bias=False)
        
        # W2 ∈ ℝ^(proj_dim × st_dim)
        # Projects interaction embeddings (node * neighbor) to latent space
        self.W2 = nn.Linear(st_dim, proj_dim, bias=False)
        
        # Bias term b ∈ ℝ^(proj_dim)
        self.bias = nn.Parameter(torch.zeros(proj_dim))


# ================================================================
#  FUNCTION: aggregate_meta_path
#  PURPOSE:
#     - Aggregates multi-hop neighbor information through L layers
#     - Combines self, neighbor, and interaction features
# ================================================================
def aggregate_meta_path(works, st_embeddings, neighbors_dict, aggregator, L=2):
    # Number of nodes
    n = len(works)
    
    # Device (CPU or GPU)
    device = st_embeddings.device
    
    # st_dim: input embedding dimension (e.g., 768 from SentenceTransformer)
    st_dim = st_embeddings.size(1)
    
    # proj_dim: projected dimension (e.g., 64)
    proj_dim = aggregator.W1.out_features

    # --------------------------------------------------------------
    # Create list of neighbor indices for each node
    # Each element is a tensor of indices of node i's neighbors
    # --------------------------------------------------------------
    neighbor_idx_list = [
        torch.tensor(sorted(list(neighbors_dict.get(i, set()))),
                     dtype=torch.long, device=device)
        for i in range(n)
    ]

    # Initialize center embeddings as input embeddings (x_i)
    center = st_embeddings

    # Store the output from each layer (multi-hop)
    order_embeddings = []

    # ===============================================================
    # Perform L-hop aggregation (each hop = one layer)
    # ===============================================================
    for _ in range(L):
        # Output tensor for this layer → (n × proj_dim)
        out_layer = torch.zeros((n, proj_dim), device=device)
        
        # Access linear layers for readability
        W1 = aggregator.W1
        W2 = aggregator.W2
        
        # Precompute self-projection for all nodes
        # h_i = W1 x_i   → shape (n × proj_dim)
        W1_center = W1(center)

        # ===========================================================
        # Iterate over all nodes
        # ===========================================================
        for i in range(n):
            # ci ∈ ℝ^(st_dim): embedding of current node i (x_i)
            ci = center[i]

            # base = W1 x_i  (node’s self representation)
            base = W1_center[i]

            # Retrieve neighbor indices
            neigh_idx = neighbor_idx_list[i]

            # Case 1: No neighbors → zero aggregation
            if neigh_idx.numel() == 0:
                agg = torch.zeros_like(base)

            # Case 2: Has neighbors
            else:
                # Neighbor embeddings x_j ∈ ℝ^(k × st_dim)
                neigh_vecs = st_embeddings[neigh_idx]

                # ------------------------------------------------------
                # w1_neigh = W1 x_j
                # Simple linear transformation of neighbor embeddings
                # shape: (k × proj_dim)
                # ------------------------------------------------------
                w1_neigh = W1(neigh_vecs)

                # ------------------------------------------------------
                # w2_part = W2 (x_j * x_i)
                # Elementwise multiplication (interaction between node & neighbors)
                # Then projected to latent space
                # shape: (k × proj_dim)
                # ------------------------------------------------------
                w2_part = W2(neigh_vecs * ci.unsqueeze(0))

                # ------------------------------------------------------
                # Sum over all neighbors:
                # agg = Σ_j (W1 x_j + W2 (x_j * x_i))
                # shape: (proj_dim)
                # ------------------------------------------------------
                agg = (w1_neigh + w2_part).sum(dim=0)

            # ----------------------------------------------------------
            # Combine self + neighbor aggregation + bias
            # out_layer[i] = W1 x_i + Σ_j [W1 x_j + W2(x_j * x_i)] + b
            # ----------------------------------------------------------
            out_layer[i] = base + agg + aggregator.bias

        # Apply non-linearity
        # ReLU(z) = max(0, z)
        out_layer = F.relu(out_layer)

        # Save this layer’s embeddings
        order_embeddings.append(out_layer)

        # --------------------------------------------------------------
        # Back-projection step:
        # Randomly project back to semantic dimension to allow next layer
        # center = tanh(out_layer × W_back)
        # --------------------------------------------------------------
        W_back = torch.randn((proj_dim, st_dim), device=device) * (1.0 / (proj_dim ** 0.5))
        center = torch.tanh(out_layer @ W_back)

    # Concatenate embeddings from all layers (multi-hop)
    # Final output: (n × (proj_dim × L))
    return torch.cat(order_embeddings, dim=1)



# ===============================================================
# MATHEMATICAL SUMMARY
# ===============================================================

# Let:
#   x_i ∈ ℝ^(st_dim)          = current node’s embedding
#   N(i)                      = set of neighbors of node i
#   W1, W2 ∈ ℝ^(proj_dim × st_dim) = learnable linear transformations
#   b ∈ ℝ^(proj_dim)          = bias term
#   L                         = number of layers (meta-path hops)

# ----------------------------------------------------------------
# 1️⃣ Self-node projection:
#       h_i = W1 x_i
#
# ----------------------------------------------------------------
# 2️⃣ Neighbor aggregation:
#       agg_i = Σ_{j ∈ N(i)} [ W1 x_j + W2 (x_j ⊙ x_i) ]
#
#       where (x_j ⊙ x_i) = elementwise product capturing
#       pairwise interaction between node i and neighbor j.
#
# ----------------------------------------------------------------
# 3️⃣ Node update:
#       z_i = ReLU(h_i + agg_i + b)
#
# ----------------------------------------------------------------
# 4️⃣ Back projection to input space:
#       x_i' = tanh(z_i W_back)
#
# ----------------------------------------------------------------
# 5️⃣ Repeat steps (1–4) for L layers.
#
# ----------------------------------------------------------------
# 6️⃣ Final multi-hop output:
#       Output_i = [ z_i^(1) ∥ z_i^(2) ∥ ... ∥ z_i^(L) ]
#
#       (concatenation of all layer representations)
#
# ----------------------------------------------------------------
# INTUITION:
#   - W1 learns how to project embeddings individually.
#   - W2 learns how to project pairwise (interaction-based) relations.
#   - The model accumulates information from neighbors up to L hops.
#   - The elementwise product (x_j ⊙ x_i) makes the aggregation
#     context-aware, emphasizing neighbors similar to the node.
# ===============================================================
