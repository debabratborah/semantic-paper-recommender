import torch
import torch.nn.functional as F
from torch import nn


class SemanticFusion(nn.Module):

    def __init__(self, input_dim, attn_dim):
        super().__init__()

        self.W = nn.Linear(input_dim, attn_dim)

        self.q = nn.Parameter(torch.randn(attn_dim))

        self.dropout = nn.Dropout(0.2)

    def forward(self, semantic_embs):

        stacked = torch.stack(semantic_embs, dim=1)

        n,K,d = stacked.shape

        proj = torch.tanh(self.W(stacked.view(n*K,d))).view(n,K,-1)

        scores = torch.matmul(proj,self.q)

        weights = F.softmax(scores,dim=1)

        fused = (weights.unsqueeze(-1)*stacked).sum(dim=1)

        return self.dropout(fused), weights