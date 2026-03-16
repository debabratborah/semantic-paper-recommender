import requests
import json
import time
from tqdm import tqdm

BASE_URL = "https://api.openalex.org/works"

# topics to download
TOPICS = [
    "machine learning",
    "deep learning",
    "graph neural networks",
    "natural language processing",
    "computer vision"
]

PER_PAGE = 200
PAGES_PER_TOPIC = 10   # 2000 papers per topic


def download_topic(topic):

    papers = []

    for page in tqdm(range(1, PAGES_PER_TOPIC + 1), desc=f"Downloading {topic}"):

        params = {
            "search": topic,
            "per_page": PER_PAGE,
            "page": page
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print("API error:", response.text)
            continue

        data = response.json()

        for work in data["results"]:

            paper = {
                "paperId": work.get("id"),
                "title": work.get("title"),

                "authors": [
                    {"name": a["author"]["display_name"]}
                    for a in work.get("authorships", [])
                    if a.get("author")
                ],

                "year": work.get("publication_year"),

                "venue": (
                    work.get("host_venue", {}).get("display_name")
                    if work.get("host_venue")
                    else None
                ),

                "concepts": [
                    c["display_name"]
                    for c in work.get("concepts", [])
                ],

                "references": work.get("referenced_works", []),

                "abstract": work.get("abstract_inverted_index", None)
            }

            papers.append(paper)

        time.sleep(1)

    return papers


def build_dataset():

    print("Starting dataset download...\n")

    all_papers = []

    for topic in TOPICS:

        print("\nTopic:", topic)

        papers = download_topic(topic)

        all_papers.extend(papers)

        print("Total collected so far:", len(all_papers))

    with open("papers.json", "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2)

    print("\nDataset saved as papers.json")
    print("Total papers downloaded:", len(all_papers))


if __name__ == "__main__":
    build_dataset()