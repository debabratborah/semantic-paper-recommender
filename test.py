import json
import random
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from config import params
from data_utils import build_meta_path_adjs
from models.aggregator import SemanticAggregator, aggregate_meta_path
from models.fusion import SemanticFusion


# =========================================================
# SETTINGS
# =========================================================

NUM_TEST_PAPERS = 100
TOP_K = 10


# =========================================================
# LOAD DATASET
# =========================================================

with open("papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"\nLoaded Papers: {len(papers)}")


# =========================================================
# FILTER VALID PAPERS
# =========================================================

papers = [
    p for p in papers
    if p.get("title")
]

random.shuffle(papers)

test_papers = papers[:NUM_TEST_PAPERS]


# =========================================================
# LOAD SENTENCE TRANSFORMER
# =========================================================

device = params["device"]

st_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# BUILD TEXTS
# =========================================================

texts = []

for p in papers:

    title = p.get("title", "") or ""

    abstract = ""

    if p.get("abstract"):
        abstract = str(p.get("abstract"))

    texts.append(title + " " + abstract)


# =========================================================
# SEMANTIC EMBEDDINGS
# =========================================================

print("\nGenerating semantic embeddings...")

paper_emb = torch.tensor(
    st_model.encode(texts),
    dtype=torch.float,
    device=device
)

paper_emb = F.normalize(
    paper_emb,
    dim=1
)


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

) = build_meta_path_adjs(papers)


# =========================================================
# LOAD HGNN MODEL
# =========================================================

fused_dim = (
    params["semantic_proj_dim"]
    * params["L"]
)

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

proj_identity = torch.nn.Linear(
    params["st_dim"],
    fused_dim
).to(device)

fusion = SemanticFusion(
    fused_dim,
    params["attn_dim"]
).to(device)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

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


# =========================================================
# HGNN AGGREGATION
# =========================================================

print("\nRunning HGNN aggregation...")

E_pap = aggregate_meta_path(
    papers,
    paper_emb,
    pap_neighbors,
    ag_pap,
    L=params["L"]
)

E_pvp = aggregate_meta_path(
    papers,
    paper_emb,
    pvp_neighbors,
    ag_pvp,
    L=params["L"]
)

E_pyp = aggregate_meta_path(
    papers,
    paper_emb,
    pyp_neighbors,
    ag_pyp,
    L=params["L"]
)

E_pkp = aggregate_meta_path(
    papers,
    paper_emb,
    pkp_neighbors,
    ag_pkp,
    L=params["L"]
)

E_pcp = aggregate_meta_path(
    papers,
    paper_emb,
    pcp_neighbors,
    ag_pcp,
    L=params["L"]
)

E_identity = torch.relu(
    proj_identity(paper_emb)
)


# =========================================================
# FUSION
# =========================================================

print("\nPerforming semantic fusion...")

fused_items, _ = fusion([

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
# DIVERSITY FUNCTION
# =========================================================

def compute_diversity(indices):

    if len(indices) <= 1:
        return 0

    total = 0
    count = 0

    for i in range(len(indices)):

        for j in range(i + 1, len(indices)):

            emb1 = paper_emb[indices[i]]
            emb2 = paper_emb[indices[j]]

            similarity = torch.dot(
                emb1,
                emb2
            ).item()

            total += similarity
            count += 1

    avg_similarity = total / count

    diversity = 1 - avg_similarity

    return diversity


# =========================================================
# METRIC STORAGE
# =========================================================

semantic_similarity_scores = []
hgnn_similarity_scores = []

semantic_diversity_scores = []
hgnn_diversity_scores = []

novelty_scores = []
overlap_scores = []
serendipity_scores = []


# =========================================================
# EVALUATION LOOP
# =========================================================

print("\nStarting evaluation...\n")

for test_paper in test_papers:

    title = test_paper.get("title", "")

    abstract = str(
        test_paper.get("abstract", "")
    )

    query = title + " " + abstract


    # =====================================================
    # QUERY EMBEDDING
    # =====================================================

    query_emb = torch.tensor(
        st_model.encode([query])[0],
        dtype=torch.float,
        device=device
    )

    query_emb = F.normalize(
        query_emb.unsqueeze(0),
        dim=1
    )


    # =====================================================
    # SEMANTIC RETRIEVAL
    # =====================================================

    semantic_scores = torch.mv(
        paper_emb,
        query_emb.squeeze()
    )

    semantic_topk = torch.topk(
        semantic_scores,
        k=min(TOP_K, len(papers))
    )

    semantic_indices = semantic_topk.indices.tolist()


    # =====================================================
    # HGNN RETRIEVAL
    # =====================================================

    q_proj = torch.relu(
        proj_identity(query_emb)
    )

    q_proj = F.normalize(
        q_proj,
        dim=1
    )

    hgnn_scores = torch.mv(
        fused_items,
        q_proj.squeeze()
    )

    hgnn_topk = torch.topk(
        hgnn_scores,
        k=min(TOP_K, len(papers))
    )

    hgnn_indices = hgnn_topk.indices.tolist()


    # =====================================================
    # AVG SIMILARITY
    # =====================================================

    semantic_avg = torch.mean(
        semantic_topk.values
    ).item()

    hgnn_avg = 0

    for idx in hgnn_indices:

        sim = torch.dot(
            query_emb.squeeze(),
            paper_emb[idx]
        ).item()

        hgnn_avg += sim

    hgnn_avg /= len(hgnn_indices)


    # =====================================================
    # DIVERSITY
    # =====================================================

    semantic_div = compute_diversity(
        semantic_indices
    )

    hgnn_div = compute_diversity(
        hgnn_indices
    )


    # =====================================================
    # OVERLAP
    # =====================================================

    semantic_set = set(semantic_indices)
    hgnn_set = set(hgnn_indices)

    overlap = len(
        semantic_set.intersection(hgnn_set)
    ) / TOP_K


    # =====================================================
    # NOVELTY
    # =====================================================

    novelty = 1 - overlap


    # =====================================================
    # SERENDIPITY
    # =====================================================

    new_items = hgnn_set - semantic_set

    serendipity = len(new_items) / TOP_K


    # =====================================================
    # STORE METRICS
    # =====================================================

    semantic_similarity_scores.append(
        semantic_avg
    )

    hgnn_similarity_scores.append(
        hgnn_avg
    )

    semantic_diversity_scores.append(
        semantic_div
    )

    hgnn_diversity_scores.append(
        hgnn_div
    )

    novelty_scores.append(
        novelty
    )

    overlap_scores.append(
        overlap
    )

    serendipity_scores.append(
        serendipity
    )


# =========================================================
# FINAL AVERAGES
# =========================================================

final_semantic_similarity = sum(
    semantic_similarity_scores
) / len(semantic_similarity_scores)

final_hgnn_similarity = sum(
    hgnn_similarity_scores
) / len(hgnn_similarity_scores)

final_semantic_diversity = sum(
    semantic_diversity_scores
) / len(semantic_diversity_scores)

final_hgnn_diversity = sum(
    hgnn_diversity_scores
) / len(hgnn_diversity_scores)

final_overlap = sum(
    overlap_scores
) / len(overlap_scores)

final_novelty = sum(
    novelty_scores
) / len(novelty_scores)

final_serendipity = sum(
    serendipity_scores
) / len(serendipity_scores)


# =========================================================
# FINAL RESULTS TABLE
# =========================================================

print("\n")
print("=" * 70)
print("FINAL HGNN EVALUATION RESULTS")
print("=" * 70)

print(
    f"{'Metric':<30}"
    f"{'Semantic':<15}"
    f"{'HGNN':<15}"
)

print("-" * 70)

print(
    f"{'Average Similarity':<30} "
    f"{final_semantic_similarity:<15.4f} "
    f"{final_hgnn_similarity:<15.4f}"
)

print(
    f"{'Diversity':<30} "
    f"{final_semantic_diversity:<15.4f} "
    f"{final_hgnn_diversity:<15.4f}"
)

print(
    f"{'Overlap Ratio':<30} "
    f"{'-':<15} "
    f"{final_overlap:<15.4f}"
)

print(
    f"{'Novelty Score':<30} "
    f"{'-':<15} "
    f"{final_novelty:<15.4f}"
)

print(
    f"{'Serendipity Score':<30} "
    f"{'-':<15} "
    f"{final_serendipity:<15.4f}"
)

print("=" * 70)
print("Evaluation Complete")
print("=" * 70)