import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAggregator(nn.Module):

    def __init__(self, st_dim, proj_dim):
        super().__init__()

        self.W1 = nn.Linear(st_dim, proj_dim, bias=False)
        self.W2 = nn.Linear(st_dim, proj_dim, bias=False)

        self.bias = nn.Parameter(torch.zeros(proj_dim))

        self.W_back = nn.Linear(proj_dim, st_dim)


def aggregate_meta_path(works, st_embeddings, neighbors_dict, aggregator, L=2):

    n = len(works)
    device = st_embeddings.device

    proj_dim = aggregator.W1.out_features

    neighbor_idx_list = [
        torch.tensor(sorted(list(neighbors_dict.get(i,set()))),
        dtype=torch.long, device=device)
        for i in range(n)
    ]

    center = st_embeddings

    order_embeddings = []

    for _ in range(L):

        out_layer = torch.zeros((n, proj_dim), device=device)

        W1 = aggregator.W1
        W2 = aggregator.W2

        W1_center = W1(center)

        for i in range(n):

            ci = center[i]

            base = W1_center[i]

            neigh_idx = neighbor_idx_list[i]

            if neigh_idx.numel()==0:

                agg = torch.zeros_like(base)

            else:

                neigh_vecs = st_embeddings[neigh_idx]

                w1_neigh = W1(neigh_vecs)

                w2_part = W2(neigh_vecs * ci.unsqueeze(0))

                agg = (w1_neigh + w2_part).sum(dim=0)

            out_layer[i] = base + agg + aggregator.bias

        out_layer = F.relu(out_layer)

        order_embeddings.append(out_layer)

        center = torch.tanh(aggregator.W_back(out_layer))

    return torch.cat(order_embeddings, dim=1)