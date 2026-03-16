import json
import os
from collections import defaultdict


# =========================================================
# LOAD PAPERS FROM LOCAL DATASET
# =========================================================

def fetch_papers(query=None, limit=50, dataset_path="papers.json"):

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset file papers.json not found")

    with open(dataset_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # if no query return first papers
    if query is None:
        return papers[:limit]

    query = query.lower()

    results = []

    for p in papers:

        title = (p.get("title") or "").lower()
        abstract = (p.get("abstract") or "").lower()

        if query in title or query in abstract:
            results.append(p)

        if len(results) >= limit:
            break

    return results


# =========================================================
# BUILD META PATH ADJACENCY LISTS
# =========================================================

def build_meta_path_adjs(works):

    author_to_papers = defaultdict(list)
    venue_to_papers = defaultdict(list)
    year_to_papers = defaultdict(list)
    keyword_to_papers = defaultdict(list)

    id_to_index = {}

    # map paperId to index
    for i, w in enumerate(works):
        pid = w.get("paperId")
        if pid:
            id_to_index[pid] = i

    # ------------------------------------------------------
    # Build node mappings
    # ------------------------------------------------------

    for i, w in enumerate(works):

        # Authors
        for a in w.get("authors", []):
            name = a.get("name")
            if name:
                author_to_papers[name].append(i)

        # Venue
        venue = w.get("venue")
        if venue:
            venue_to_papers[venue].append(i)

        # Year
        year = w.get("year")
        if year:
            year_to_papers[year].append(i)

        # Keywords / concepts
        for c in w.get("concepts", []):
            keyword_to_papers[c].append(i)

    # ------------------------------------------------------
    # Initialize neighbor graphs
    # ------------------------------------------------------

    pap_neighbors = defaultdict(set)
    pvp_neighbors = defaultdict(set)
    pyp_neighbors = defaultdict(set)
    pkp_neighbors = defaultdict(set)
    pcp_neighbors = defaultdict(set)

    # ------------------------------------------------------
    # PAP : Paper → Author → Paper
    # ------------------------------------------------------

    for papers in author_to_papers.values():
        for p in papers:
            pap_neighbors[p].update(papers)

    # ------------------------------------------------------
    # PVP : Paper → Venue → Paper
    # ------------------------------------------------------

    for papers in venue_to_papers.values():
        for p in papers:
            pvp_neighbors[p].update(papers)

    # ------------------------------------------------------
    # PYP : Paper → Year → Paper
    # ------------------------------------------------------

    for papers in year_to_papers.values():
        for p in papers:
            pyp_neighbors[p].update(papers)

    # ------------------------------------------------------
    # PKP : Paper → Keyword → Paper
    # ------------------------------------------------------

    for papers in keyword_to_papers.values():
        for p in papers:
            pkp_neighbors[p].update(papers)

    # ------------------------------------------------------
    # PCP : Paper → Citation → Paper
    # ------------------------------------------------------

    for i, w in enumerate(works):

        for ref in w.get("references", []):

            if ref in id_to_index:
                j = id_to_index[ref]

                pcp_neighbors[i].add(j)
                pcp_neighbors[j].add(i)

    # ------------------------------------------------------
    # Remove self loops
    # ------------------------------------------------------

    for p in range(len(works)):

        pap_neighbors[p].discard(p)
        pvp_neighbors[p].discard(p)
        pyp_neighbors[p].discard(p)
        pkp_neighbors[p].discard(p)
        pcp_neighbors[p].discard(p)

    return (
        pap_neighbors,
        pvp_neighbors,
        pyp_neighbors,
        pkp_neighbors,
        pcp_neighbors
    )