import json
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DATASET_PATH = "papers.json"
OUTPUT_PATH = "paper_embeddings.pt"


def reconstruct_abstract(inv_index):

    if inv_index is None:
        return ""

    words = []

    for word, positions in inv_index.items():
        for pos in positions:
            words.append((pos, word))

    words.sort()

    return " ".join([w for _, w in words])


def generate_embeddings():

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        works = json.load(f)

    print("Loaded papers:", len(works))

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = []

    for w in works:

        title = w.get("title") or ""

        abstract = w.get("abstract")

        if isinstance(abstract, dict):
            abstract = reconstruct_abstract(abstract)

        if abstract is None:
            abstract = ""

        texts.append(title + " " + abstract)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    embeddings = torch.tensor(embeddings, dtype=torch.float)

    torch.save(embeddings, OUTPUT_PATH)

    print("Embeddings saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    generate_embeddings()