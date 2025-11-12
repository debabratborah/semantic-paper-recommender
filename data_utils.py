import requests
import json
import os
from collections import defaultdict

def fetch_papers(query, limit=25, cache_file="cached_works.json"):
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if query in cached:
            print(f"[DEBUG] Using cached results for query: '{query}'")
            return cached[query]

    print(f"[DEBUG] Fetching papers from Semantic Scholar for query: '{query}'")
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,abstract,authors,venue,url,year"
    params_req = {"query": query, "limit": limit, "fields": fields}
    resp = requests.get(base, params=params_req)
    resp.raise_for_status()
    works = resp.json().get("data", [])

    cached = {query: works}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f, indent=2)
    return works


def build_meta_path_adjs(works):
    """Builds adjacency lists for P-A-P and P-V-P meta-paths."""
    author_to_papers = defaultdict(list)
    venue_to_papers = defaultdict(list)

    for i, w in enumerate(works):
        for a in w.get("authors", []):
            if a.get("name"):
                author_to_papers[a["name"]].append(i)
        v = w.get("venue")
        if v:
            venue_to_papers[v].append(i)

    pap_neighbors = defaultdict(set)
    pvp_neighbors = defaultdict(set)

    for author, paper_list in author_to_papers.items():
        for p in paper_list:
            pap_neighbors[p].update(paper_list)

    for venue, paper_list in venue_to_papers.items():
        for p in paper_list:
            pvp_neighbors[p].update(paper_list)

    for p in range(len(works)):
        pap_neighbors[p].discard(p)
        pvp_neighbors[p].discard(p)

    return pap_neighbors, pvp_neighbors
