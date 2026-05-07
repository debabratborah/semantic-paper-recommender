import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer

from config import params
from data_utils import reconstruct_abstract
from models.aggregator import SemanticAggregator
from models.fusion import SemanticFusion


def load_precomputed(device):
    data = torch.load("fused_embeddings.pt", map_location=device)
    fused_embeddings = data["fused_embeddings"]
    works = data["works"]
    return fused_embeddings, works


def load_proj_identity(device):
    fused_dim = params["semantic_proj_dim"] * params["L"]
    proj_identity = nn.Linear(params["st_dim"], fused_dim).to(device)
    ckpt = torch.load("model_trained.pt", map_location=device)
    proj_identity.load_state_dict(ckpt["proj_identity"])
    return proj_identity


def load_model(device):
    fused_dim = params["semantic_proj_dim"] * params["L"]

    ag_pap = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pvp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pyp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pkp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pcp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    proj_identity = nn.Linear(params["st_dim"], fused_dim).to(device)
    fusion = SemanticFusion(fused_dim, params["attn_dim"]).to(device)

    ckpt = torch.load("model_trained.pt", map_location=device)
    ag_pap.load_state_dict(ckpt["aggregator_pap"])
    ag_pvp.load_state_dict(ckpt["aggregator_pvp"])
    ag_pyp.load_state_dict(ckpt["aggregator_pyp"])
    ag_pkp.load_state_dict(ckpt["aggregator_pkp"])
    ag_pcp.load_state_dict(ckpt["aggregator_pcp"])
    proj_identity.load_state_dict(ckpt["proj_identity"])
    fusion.load_state_dict(ckpt["fusion"])

    return ag_pap, ag_pvp, ag_pyp, ag_pkp, ag_pcp, proj_identity, fusion


def recommend(query, limit=25, top_k=10):
    device = params["device"]

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    proj_identity = load_proj_identity(device)

    fused_embeddings, works = load_precomputed(device)

    q_emb = torch.tensor(
        st_model.encode([query])[0], dtype=torch.float, device=device
    )
    q_proj = F.normalize(torch.relu(proj_identity(q_emb)), dim=0)

    scores = torch.mv(fused_embeddings, q_proj)
    k = min(top_k, len(scores))
    topk = torch.topk(scores, k=k)

    recs = []
    for rank, (idx, score) in enumerate(
        zip(topk.indices.tolist(), topk.values.tolist()), start=1
    ):
        p = works[idx]
        recs.append({
            "rank":    rank,
            "title":   p.get("title", "N/A"),
            "authors": [a.get("name") for a in p.get("authors", [])][:5],
            "venue":   p.get("venue", "N/A"),
            "year":    p.get("year", "N/A"),
            "url":     p.get("url", "N/A"),
            "score":   float(score),
        })

    return recs, None

