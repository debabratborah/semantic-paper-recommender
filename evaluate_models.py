import torch
import torch.nn.functional as F
from hgrec_recommendation import HGRecommender
from hgrec_minus import HGRecMinusRecommender

# -------------------------
# Evaluation Metrics
# -------------------------

def precision_at_k(ranked_indices, relevant_indices, k):
    ranked = ranked_indices[:k]
    hits = sum([1 for i in ranked if i in relevant_indices])
    return hits / k

def recall_at_k(ranked_indices, relevant_indices, k):
    ranked = ranked_indices[:k]
    hits = sum([1 for i in ranked if i in relevant_indices])
    return hits / len(relevant_indices) if relevant_indices else 0.0

def ndcg_at_k(ranked_indices, relevant_indices, k):
    ranked = ranked_indices[:k]
    dcg = 0.0
    for i, idx in enumerate(ranked):
        if idx in relevant_indices:
            dcg += 1.0 / torch.log2(torch.tensor(i + 2.0))  # log base 2
    # ideal DCG
    idcg = sum(1.0 / torch.log2(torch.tensor(i + 2.0)) for i in range(min(len(relevant_indices), k)))
    return (dcg / idcg).item() if idcg > 0 else 0.0

# -------------------------
# Evaluation Function
# -------------------------

def evaluate_model(recommender_class, query, k=5):
    recommender = recommender_class(query)
    recommender.train()

    # Encode query
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("intfloat/e5-base-v2")
    query_embedding = embed_model.encode(query, convert_to_tensor=True).detach().cpu()

    # All paper embeddings
    paper_embeddings = recommender.data["paper"].x.detach().cpu()

    # Cosine similarity ranking
    sims = F.cosine_similarity(query_embedding.unsqueeze(0), paper_embeddings)
    ranked_indices = torch.argsort(sims, descending=True).tolist()

    # Relevance ground-truth: here, if query word in title → relevant
    relevant_indices = [i for i, title in enumerate(recommender.titles) if query.lower() in title.lower()]

    metrics = {
        "Precision@K": precision_at_k(ranked_indices, relevant_indices, k),
        "Recall@K": recall_at_k(ranked_indices, relevant_indices, k),
        "NDCG@K": ndcg_at_k(ranked_indices, relevant_indices, k),
    }
    return metrics

# -------------------------
# Run Evaluation
# -------------------------

def main():
    queries = ["heterogeneous graph", "graph neural network", "recommendation system"]
    models = {
        "HGRec": HGRecommender,
        "HGRec-": HGRecMinusRecommender,
    }

    k = 5
    for query in queries:
        print(f"\n===== Query: {query} =====")
        for model_name, model_class in models.items():
            metrics = evaluate_model(model_class, query, k=k)
            print(f"[{model_name}] Precision@{k}: {metrics['Precision@K']:.3f}, "
                  f"Recall@{k}: {metrics['Recall@K']:.3f}, "
                  f"NDCG@{k}: {metrics['NDCG@K']:.3f}")

if __name__ == "__main__":
    main()
