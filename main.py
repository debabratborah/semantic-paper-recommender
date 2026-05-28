import os
import json
import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer

from config import params
from data_utils import fetch_papers, build_meta_path_adjs
from models.aggregator import SemanticAggregator, aggregate_meta_path
from models.fusion import SemanticFusion


# =========================================================
# LOAD SENTENCE TRANSFORMER
# =========================================================

def load_sentence_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# MAIN RECOMMENDATION FUNCTION
# =========================================================

def get_recommendations(query, top_k):

    device = params["device"]

    st_model = load_sentence_model()

    # =========================================================
    # INTERNAL CANDIDATE SIZE
    # =========================================================

    candidate_size = 200

    # =========================================================
    # SEARCH WHOLE DATASET
    # =========================================================

    works = fetch_papers(
        query=query,
        limit=candidate_size
    )

    if not works:

        print("No papers found.")

        return None

    print("\nLoaded Candidate Papers:", len(works))

    # =========================================================
    # BUILD TEXTS
    # =========================================================

    texts = []

    for w in works:

        title = w.get("title", "") or ""

        abstract = ""

        if w.get("abstract"):

            abstract = str(w.get("abstract"))

        texts.append(
            title + " " + abstract
        )

    # =========================================================
    # GENERATE SEMANTIC EMBEDDINGS
    # =========================================================

    print("\nGenerating semantic embeddings...")

    paper_emb = torch.tensor(
        st_model.encode(texts),
        dtype=torch.float,
        device=device
    )

    paper_emb_norm = F.normalize(
        paper_emb,
        dim=1
    )

    # =========================================================
    # QUERY EMBEDDING
    # =========================================================

    query_emb = torch.tensor(
        st_model.encode([query])[0],
        dtype=torch.float,
        device=device
    )

    query_emb = F.normalize(
        query_emb.unsqueeze(0),
        dim=1
    )

    # =========================================================
    # SEMANTIC RETRIEVAL
    # =========================================================

    semantic_scores = torch.mv(
        paper_emb_norm,
        query_emb.squeeze()
    )

    semantic_topk = torch.topk(
        semantic_scores,
        k=min(top_k, len(works))
    )

    semantic_recs = []

    print("\n===================================================")
    print("SEMANTIC RETRIEVAL RESULTS")
    print("===================================================")

    for rank, (idx, score) in enumerate(

        zip(
            semantic_topk.indices.tolist(),
            semantic_topk.values.tolist()
        ),

        start=1
    ):

        p = works[idx]

        semantic_recs.append({

            "rank": rank,

            "title": p.get("title", "N/A"),

            "score": float(score),

            "paperId": p.get("paperId", None)

        })

        print(f"\nRank: {rank}")
        print(f"Title: {p.get('title', 'N/A')}")
        print(f"Score: {float(score):.4f}")

    # =========================================================
    # BUILD HETEROGENEOUS GRAPH
    # =========================================================

    print("\nBuilding heterogeneous graph...")

    (
        pap_neighbors,
        pvp_neighbors,
        pyp_neighbors,
        pkp_neighbors,
        pcp_neighbors

    ) = build_meta_path_adjs(works)

    fused_dim = (
        params["semantic_proj_dim"]
        * params["L"]
    )

    # =========================================================
    # LOAD HGNN MODULES
    # =========================================================

    ag_pap = SemanticAggregator(
        params["st_dim"],
        params["semantic_proj_dim"]
    ).to(device)

    ag_pvp = SemanticAggregator(
        params["st_dim"],
        params["semantic_proj_dim"]
    ).to(device)

    ag_pyp = SemanticAggregator(
        params["st_dim"],
        params["semantic_proj_dim"]
    ).to(device)

    ag_pkp = SemanticAggregator(
        params["st_dim"],
        params["semantic_proj_dim"]
    ).to(device)

    ag_pcp = SemanticAggregator(
        params["st_dim"],
        params["semantic_proj_dim"]
    ).to(device)

    proj_identity = nn.Linear(
        params["st_dim"],
        fused_dim
    ).to(device)

    fusion = SemanticFusion(
        fused_dim,
        params["attn_dim"]
    ).to(device)

    trained = False

    # =========================================================
    # LOAD TRAINED MODEL
    # =========================================================

    if os.path.exists("model_trained.pt"):

        print("\nLoading trained HGNN model...")

        ckpt = torch.load(
            "model_trained.pt",
            map_location=device
        )

        ag_pap.load_state_dict(
            ckpt["aggregator_pap"]
        )

        ag_pvp.load_state_dict(
            ckpt["aggregator_pvp"]
        )

        ag_pyp.load_state_dict(
            ckpt["aggregator_pyp"]
        )

        ag_pkp.load_state_dict(
            ckpt["aggregator_pkp"]
        )

        ag_pcp.load_state_dict(
            ckpt["aggregator_pcp"]
        )

        proj_identity.load_state_dict(
            ckpt["proj_identity"]
        )

        fusion.load_state_dict(
            ckpt["fusion"]
        )

        trained = True

    else:

        print("\nWARNING: model_trained.pt not found")

    # =========================================================
    # HGNN AGGREGATION
    # =========================================================

    print("\nPerforming HGNN aggregation...")

    E_pap = aggregate_meta_path(
        works,
        paper_emb,
        pap_neighbors,
        ag_pap,
        L=params["L"]
    )

    E_pvp = aggregate_meta_path(
        works,
        paper_emb,
        pvp_neighbors,
        ag_pvp,
        L=params["L"]
    )

    E_pyp = aggregate_meta_path(
        works,
        paper_emb,
        pyp_neighbors,
        ag_pyp,
        L=params["L"]
    )

    E_pkp = aggregate_meta_path(
        works,
        paper_emb,
        pkp_neighbors,
        ag_pkp,
        L=params["L"]
    )

    E_pcp = aggregate_meta_path(
        works,
        paper_emb,
        pcp_neighbors,
        ag_pcp,
        L=params["L"]
    )

    # =========================================================
    # IDENTITY EMBEDDING
    # =========================================================

    E_identity = torch.relu(
        proj_identity(paper_emb)
    )

    # =========================================================
    # FUSION
    # =========================================================

    print("\nPerforming semantic fusion...")

    fused_items, meta_weights = fusion([

        E_identity,

        E_pap,

        E_pvp,

        E_pyp,

        E_pkp,

        E_pcp

    ])

    fused_items = F.normalize(
        fused_items,
        dim=1
    )

    # =========================================================
    # BEST SEMANTIC PAPER
    # =========================================================

    best_idx = semantic_topk.indices[0]

    matched_paper = works[best_idx].get(
        "title",
        "N/A"
    )

    print("\n===================================================")
    print("MATCHED SEMANTIC PAPER")
    print("===================================================")

    print(matched_paper)

    # =========================================================
    # HGNN QUERY REPRESENTATION
    # =========================================================

    q_proj = fused_items[best_idx]

    # =========================================================
    # HGNN RETRIEVAL
    # =========================================================

    print("\nComputing HGNN similarities...")

    hgnn_scores = torch.mv(
        fused_items,
        q_proj
    )

    hgnn_scores[best_idx] = -1e9

    hgnn_topk = torch.topk(
        hgnn_scores,
        k=min(top_k, len(works))
    )

    hgnn_recs = []

    print("\n===================================================")
    print("HGNN RECOMMENDATIONS")
    print("===================================================")

    for rank, (idx, score) in enumerate(

        zip(
            hgnn_topk.indices.tolist(),
            hgnn_topk.values.tolist()
        ),

        start=1
    ):

        p = works[idx]

        hgnn_recs.append({

            "rank": rank,

            "title": p.get("title", "N/A"),

            "score": float(score),

            "paperId": p.get("paperId", None)

        })

        print(f"\nRank: {rank}")
        print(f"Title: {p.get('title', 'N/A')}")
        print(f"Score: {float(score):.4f}")

    # =========================================================
    # OVERLAP ANALYSIS
    # =========================================================

    semantic_ids = set([

        p["paperId"]

        for p in semantic_recs

    ])

    hgnn_ids = set([

        p["paperId"]

        for p in hgnn_recs

    ])

    intersection = semantic_ids.intersection(
        hgnn_ids
    )

    union = semantic_ids.union(
        hgnn_ids
    )

    overlap_ratio = (

        len(intersection)

        / max(len(union), 1)

    )

    print("\n===================================================")
    print("OVERLAP ANALYSIS")
    print("===================================================")

    print(
        f"Overlap Ratio: "
        f"{overlap_ratio:.4f}"
    )

    # =========================================================
    # STATUS
    # =========================================================

    print("\n===================================================")

    if trained:

        print("Trained HGNN model loaded.")

    else:

        print("HGNN model NOT loaded.")

    return {

        "semantic_recommendations":
            semantic_recs,

        "hgnn_recommendations":
            hgnn_recs,

        "overlap_ratio":
            overlap_ratio

    }


# =========================================================
# MAIN
# =========================================================

def main():

    query = input(
        "\nEnter Research Query: "
    )

    top_k = int(
        input(
            "Number of recommendations: "
        )
    )

    get_recommendations(
        query,
        top_k
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()