import torch
import torch.nn.functional as F
from torch import nn

# ================================================================
#  CLASS: SemanticFusion
#  PURPOSE:
#     - Fuses multiple semantic embeddings (e.g., meta-path embeddings)
#       into a single unified representation using an attention mechanism.
#     - Learns how important each embedding (semantic view) is for a node.
# ================================================================
class SemanticFusion(nn.Module):
    def __init__(self, input_dim, attn_dim):
        super().__init__()
        
        # Linear projection: W ∈ ℝ^(attn_dim × input_dim)
        # Projects embeddings from input space into attention space.
        self.W = nn.Linear(input_dim, attn_dim, bias=True)
        
        # Global attention query vector: q ∈ ℝ^(attn_dim)
        # Learns to score the importance of each semantic embedding.
        self.q = nn.Parameter(torch.randn(attn_dim))
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.2)

    # ------------------------------------------------------------
    # FORWARD PASS
    # ------------------------------------------------------------
    def forward(self, semantic_embs):
        # semantic_embs: list of K tensors [n × d]
        # Each tensor represents embeddings for one semantic source
        # e.g., meta-path-specific embeddings for all n nodes.
        if not semantic_embs:
            return None

        # --------------------------------------------------------
        # 1️⃣ Stack the embeddings from all K sources:
        #    stacked ∈ ℝ^(n × K × d)
        # --------------------------------------------------------
        stacked = torch.stack(semantic_embs, dim=1)
        # n = number of nodes, K = number of embeddings, d = embedding dimension
        n, K, d = stacked.shape

        # --------------------------------------------------------
        # 2️⃣ Project embeddings into attention space:
        #    Apply W and tanh non-linearity to each semantic embedding.
        #    proj ∈ ℝ^(n × K × attn_dim)
        # --------------------------------------------------------
        # Flatten to apply linear layer:
        # stacked.view(n*K, d) → combine node and semantic dimensions.
        proj = torch.tanh(self.W(stacked.view(n * K, d))).view(n, K, -1)

        # --------------------------------------------------------
        # 3️⃣ Compute attention scores using the query vector q:
        #    scores[i, k] = <proj[i, k], q>
        #    → scores ∈ ℝ^(n × K)
        # --------------------------------------------------------
        scores = torch.matmul(proj, self.q)
        # This gives one scalar attention score per semantic embedding per node.

        # --------------------------------------------------------
        # 4️⃣ Normalize attention scores with softmax:
        #    α_ik = exp(s_ik) / Σ_j exp(s_ij)
        #    weights ∈ ℝ^(n × K)
        # --------------------------------------------------------
        weights = F.softmax(scores, dim=1)

        # --------------------------------------------------------
        # 5️⃣ Weighted sum (fusion) across K embeddings:
        #    fused[i] = Σ_k α_ik * x_ik
        #    fused ∈ ℝ^(n × d)
        # --------------------------------------------------------
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)

        # --------------------------------------------------------
        # 6️⃣ Apply dropout and return results:
        #    dropout prevents overfitting.
        #    Output: fused ∈ ℝ^(n × d), weights ∈ ℝ^(n × K)
        # --------------------------------------------------------
        return self.dropout(fused), weights



# ===============================================================
# MATHEMATICAL SUMMARY
# ===============================================================

# Let:
#   n = number of nodes
#   K = number of semantic sources (e.g., meta-path embeddings)
#   d = input embedding dimension
#   attn_dim = attention hidden dimension
#
#   X_i = [x_i1, x_i2, ..., x_iK]  where x_ik ∈ ℝ^d
#         (semantic embeddings for node i from K sources)
#
#   W ∈ ℝ^(attn_dim × d)    → linear projection
#   q ∈ ℝ^(attn_dim)        → learnable attention query vector
#
# ----------------------------------------------------------------
# 1️⃣ Projection to attention space:
#       h_ik = tanh(W x_ik)
#       where h_ik ∈ ℝ^(attn_dim)
#
# ----------------------------------------------------------------
# 2️⃣ Compute unnormalized attention score:
#       s_ik = ⟨h_ik, q⟩ = qᵀ h_ik
#
# ----------------------------------------------------------------
# 3️⃣ Normalize scores (softmax across K embeddings):
#       α_ik = exp(s_ik) / Σ_{j=1..K} exp(s_ij)
#
# ----------------------------------------------------------------
# 4️⃣ Weighted fusion (attention-weighted sum):
#       f_i = Σ_{k=1..K} α_ik * x_ik
#       where f_i ∈ ℝ^d  → fused embedding for node i
#
# ----------------------------------------------------------------
# 5️⃣ Apply dropout:
#       f_i' = Dropout(f_i)
#
# ----------------------------------------------------------------
# OUTPUTS:
#   - fused ∈ ℝ^(n × d): attention-weighted semantic embeddings
#   - weights ∈ ℝ^(n × K): attention coefficients (importance of each semantic source)
#
# ----------------------------------------------------------------
# INTUITION:
#   - W projects all semantic embeddings into a shared attention space.
#   - q acts as a “global query” that measures how relevant each semantic view is.
#   - Softmax converts scores into probabilities (attention weights).
#   - Weighted sum fuses multiple views into a single embedding per node.
#   - Dropout prevents over-reliance on any single semantic source.
# ===============================================================
