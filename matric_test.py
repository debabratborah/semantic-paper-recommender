# metrics.py

import numpy as np
from sklearn.metrics import ndcg_score


def precision_at_k(recommended, relevant, k):
    """
    Precision@K

    recommended : list of recommended paper IDs
    relevant    : set/list of relevant paper IDs
    """

    recommended_k = recommended[:k]

    hits = sum([1 for item in recommended_k if item in relevant])

    return hits / k


def recall_at_k(recommended, relevant, k):
    """
    Recall@K
    """

    recommended_k = recommended[:k]

    hits = sum([1 for item in recommended_k if item in relevant])

    return hits / len(relevant) if len(relevant) > 0 else 0


def average_precision(recommended, relevant):
    """
    Average Precision (AP)
    """

    score = 0.0
    hits = 0

    for i, item in enumerate(recommended):

        if item in relevant:
            hits += 1

            precision = hits / (i + 1)

            score += precision

    return score / len(relevant) if len(relevant) > 0 else 0


def mean_average_precision(all_recommended, all_relevant):
    """
    MAP = mean of AP across all queries
    """

    aps = []

    for recs, rels in zip(all_recommended, all_relevant):

        ap = average_precision(recs, rels)

        aps.append(ap)

    return np.mean(aps)


def ndcg_at_k(recommended_scores, ground_truth_scores, k):
    """
    NDCG@K

    recommended_scores :
        predicted ranking scores

    ground_truth_scores :
        actual relevance scores
    """

    recommended_scores = np.asarray([recommended_scores[:k]])

    ground_truth_scores = np.asarray([ground_truth_scores[:k]])

    return ndcg_score(
        ground_truth_scores,
        recommended_scores
    )


def hit_rate_at_k(recommended, relevant, k):
    """
    Hit Rate@K

    Returns 1 if at least one relevant item
    appears in top-k recommendations
    """

    recommended_k = recommended[:k]

    for item in recommended_k:

        if item in relevant:
            return 1

    return 0


# =========================================================
# Example Usage
# =========================================================

if __name__ == "__main__":

    # Recommended paper IDs
    recommended = [101, 102, 103, 104, 105]

    # Relevant paper IDs
    relevant = {102, 104, 110}

    k = 3

    print(f"Precision@{k}:",
          precision_at_k(recommended, relevant, k))

    print(f"Recall@{k}:",
          recall_at_k(recommended, relevant, k))

    print("Average Precision:",
          average_precision(recommended, relevant))

    print(f"Hit Rate@{k}:",
          hit_rate_at_k(recommended, relevant, k))

    # NDCG example
    predicted_scores = [0.95, 0.90, 0.80, 0.70, 0.60]

    actual_scores = [3, 2, 3, 0, 1]

    print(f"NDCG@{k}:",
          ndcg_at_k(predicted_scores, actual_scores, k))