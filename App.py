import os
import torch
import torch.nn.functional as F
from torch import nn

import streamlit as st
from streamlit.components.v1 import html
from sentence_transformers import SentenceTransformer

from config import params
from data_utils import fetch_papers, build_meta_path_adjs
from models.aggregator import SemanticAggregator, aggregate_meta_path
from models.fusion import SemanticFusion
from train import train_model
from visualize_graph import visualize_graph


# ===========================================================
# PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="MetaPath GNN Recommender",
    page_icon="🧠",
    layout="wide"
)


# ===========================================================
# CYBERPUNK THEME CSS
# ===========================================================
def inject_cyberpunk_css():
    st.markdown(
        """
    <style>
    /* --- GLOBAL --- */
    body, .block-container {
        background: radial-gradient(circle at top left, #1a1c2c 0, #050816 35%, #02010a 100%) !important;
        color: #E6EBFF !important;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    /* Remove Streamlit default white background */
    .stApp {
        background-color: transparent;
    }

    /* --- TITLE --- */
    .neon-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: linear-gradient(120deg, #00FFF0, #FF00E6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 12px rgba(0,255,240,0.5);
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #A0A3C2;
        margin-bottom: 1.5rem;
    }

    /* --- CARDS --- */
    .rec-card {
        background: linear-gradient(135deg, rgba(16,20,55,0.95), rgba(7,10,35,0.98));
        border-radius: 16px;
        padding: 16px 18px;
        border: 1px solid rgba(0,255,240,0.15);
        box-shadow:
            0 0 18px rgba(0,0,0,0.8),
            0 0 18px rgba(0,255,240,0.12);
        margin-bottom: 14px;
        transition: all 0.2s ease-out;
    }
    .rec-card:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow:
            0 0 24px rgba(0,0,0,0.9),
            0 0 24px rgba(0,255,240,0.25);
        border-color: rgba(255,0,200,0.45);
    }
    .rec-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #E8EDFF;
    }
    .rec-meta {
        color: #A4A8D0;
        font-size: 0.86rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-right: 4px;
        background: rgba(0,255,240,0.08);
        border: 1px solid rgba(0,255,240,0.35);
        color: #C5F7FF;
    }
    .score-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.75rem;
        background: rgba(255,0,200,0.1);
        border: 1px solid rgba(255,0,200,0.6);
        color: #FFD6FA;
    }
    .rec-link a {
        color: #4FD1FF !important;
        font-size: 0.9rem;
        text-decoration: none;
    }
    .rec-link a:hover {
        text-decoration: underline;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background: radial-gradient(circle at top, #15172d 0, #050816 60%);
        border-right: 1px solid rgba(0,255,240,0.2);
    }
    [data-testid="stSidebar"] * {
        color: #E6EBFF !important;
    }

    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(10, 12, 38, 0.8);
        border-radius: 999px;
        color: #A0A4D4 !important;
        padding: 0.35rem 0.9rem;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #00FFF0, #FF00E6) !important;
        color: #050816 !important;
        font-weight: 700 !important;
        border: 0 !important;
        box-shadow: 0 0 14px rgba(0,255,240,0.5);
    }

    /* --- METRIC BAR --- */
    .attn-row {
        display: flex;
        flex-direction: column;
        gap: 4px;
        margin-top: 0.6rem;
    }
    .attn-label {
        font-size: 0.85rem;
        color: #C5C8FF;
    }
    .attn-bar {
        height: 6px;
        border-radius: 999px;
        background: linear-gradient(90deg, #00FFF0, #FF00E6);
        box-shadow: 0 0 10px rgba(0,255,240,0.7);
    }

    </style>
    """,
        unsafe_allow_html=True,
    )


# ===========================================================
# CACHED SENTENCE MODEL
# ===========================================================
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("intfloat/e5-base-v2")


# ===========================================================
# RECOMMENDATION PIPELINE
# ===========================================================
def get_recommendations(query: str, limit: int, top_k: int):
    device = params["device"]
    st_model = load_sentence_model()

    works = fetch_papers(query, limit=limit)
    if not works:
        return [], None, False, works

    titles = [w.get("title", "") for w in works]
    paper_emb = torch.tensor(
        st_model.encode(titles),
        dtype=torch.float,
        device=device,
    )

    pap_neighbors, pvp_neighbors = build_meta_path_adjs(works)

    aggregator_pap = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    aggregator_pvp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    proj_identity = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * params["L"]).to(device)
    fusion = SemanticFusion(params["semantic_proj_dim"] * params["L"], params["semantic_proj_dim"]).to(device)
    proj_user = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * params["L"]).to(device)

    trained = False
    if os.path.exists("model_trained.pt"):
        ckpt = torch.load("model_trained.pt", map_location=device)
        aggregator_pap.load_state_dict(ckpt["aggregator_pap"])
        aggregator_pvp.load_state_dict(ckpt["aggregator_pvp"])
        proj_identity.load_state_dict(ckpt["proj_identity"])
        fusion.load_state_dict(ckpt["fusion"])
        proj_user.load_state_dict(ckpt["proj_user"])
        trained = True

    # Meta-path aggregation
    E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, aggregator_pap, L=params["L"])
    E_pvp = aggregate_meta_path(works, paper_emb, pvp_neighbors, aggregator_pvp, L=params["L"])
    E_identity = F.relu(proj_identity(paper_emb))

    fused_items, meta_weights = fusion([E_identity, E_pap, E_pvp])
    fused_items = F.normalize(fused_items, dim=1)

    q_emb = torch.tensor(
        st_model.encode([query])[0],
        dtype=torch.float,
        device=device,
    )
    q_proj = F.normalize(proj_user(q_emb), dim=0)

    scores = torch.mv(fused_items, q_proj)
    k = min(top_k, scores.size(0))
    topk = torch.topk(scores, k=k)

    recs = []
    for rank, (idx, sc) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
        paper = works[idx]
        recs.append(
            {
                "rank": rank,
                "score": float(sc),
                "title": paper.get("title", "N/A"),
                "authors": [a.get("name") for a in paper.get("authors", [])][:5],
                "venue": paper.get("venue", "N/A"),
                "year": paper.get("year", "N/A"),
                "url": paper.get("url", "N/A"),
                "index": idx,
            }
        )

    return recs, meta_weights, trained, works


# ===========================================================
# MAIN APP
# ===========================================================
def main():
    inject_cyberpunk_css()

    # Header
    st.markdown(
        "<div class='neon-title'>Meta-Path GNN Paper Recommender</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>Semantic + Heterogeneous Graph Intelligence • Papers • Authors • Venues</div>",
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("⚙️ Control Panel")
    query = st.text_input(
        "🔍 Enter research topic:",
        value="Neural Network",
        placeholder="e.g., Graph Neural Networks for Recommendation",
    )
    limit = st.sidebar.slider("Number of papers to fetch", 10, 60, 30, step=5)
    top_k = st.sidebar.slider("Top-K recommendations", 3, 20, 10, step=1)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"💻 **Device:** `{params['device']}`")

    tabs = st.tabs(["📚 Recommendations", "🌐 Graph View", "🧪 Train Model", "🧠 Model Details"])

    # ------------------------------------------------------ TAB 1 — RECOMMEND
    with tabs[0]:
        st.subheader("📚 Recommended Papers")

        if st.button("⚡ Generate Recommendations"):
            if not query.strip():
                st.warning("Please enter a query first.")
            else:
                with st.spinner("Running meta-path GNN and attention fusion..."):
                    recs, meta_weights, is_trained, works = get_recommendations(query, limit, top_k)

                if not recs:
                    st.error("No papers fetched for this query.")
                else:
                    if is_trained:
                        st.success("✅ Using **trained model** (BPR).")
                    else:
                        st.warning("⚠ Model not trained yet — using random-initialized weights.")

                    for r in recs:
                        authors = ", ".join(a for a in r["authors"] if a) or "N/A"
                        st.markdown(
                            f"""
                            <div class="rec-card">
                                <div class="rec-title">{r['rank']}. {r['title']}</div>
                                <div class="rec-meta"><b>Authors:</b> {authors}</div>
                                <div class="rec-meta"><b>Venue:</b> {r['venue']} &nbsp;|&nbsp; <b>Year:</b> {r['year']}</div>
                                <div style="margin-top:6px;">
                                    <span class="score-tag">Score: {r['score']:.4f}</span>
                                </div>
                                <div class="rec-link" style="margin-top:6px;">
                                    {'<a href="'+r['url']+'" target="_blank">🔗 Open Paper</a>' if r['url'] and r['url']!='N/A' else ''}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if meta_weights is not None and recs:
                        st.markdown("### 🔬 Meta-Path Attention for Top Paper")
                        top_idx = recs[0]["index"]
                        w = meta_weights[top_idx].detach().cpu().numpy()
                        labels = ["Identity (Text)", "PAP (Author)", "PVP (Venue)"]

                        attn_cols = st.columns(3)
                        for col, name, val in zip(attn_cols, labels, w):
                            with col:
                                st.markdown(f"<div class='attn-label'>{name}</div>", unsafe_allow_html=True)
                                st.markdown(
                                    f"<div class='attn-bar' style='width:{max(5, val*100)}%; opacity:0.9;'></div>",
                                    unsafe_allow_html=True,
                                )
                                st.write(f"`{val:.4f}`")

    # ------------------------------------------------------ TAB 2 — GRAPH VIEW
    with tabs[1]:
        st.subheader("🌐 Heterogeneous Graph Visualization")

        st.write(
            "Visualize the local graph of **papers**, **authors**, and **venues** "
            "with meta-path edges (PAP in green, PVP in red)."
        )

        if st.button("🌎 Generate & View Graph"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Fetching papers and building heterogeneous graph..."):
                    works = fetch_papers(query, limit=limit)
                    if not works:
                        st.error("No papers fetched for this query.")
                    else:
                        pap_neighbors, pvp_neighbors = build_meta_path_adjs(works)
                        visualize_graph(works, pap_neighbors, pvp_neighbors)

                if os.path.exists("graph.html"):
                    with open("graph.html", "r", encoding="utf-8") as f:
                        graph_html = f.read()
                    html(graph_html, height=800, scrolling=True)
                else:
                    st.error("graph.html not found. Please check visualize_graph implementation.")

    # ------------------------------------------------------ TAB 3 — TRAIN MODEL
    with tabs[2]:
        st.subheader("🧪 Train Meta-Path GNN with BPR Loss")
        st.write(
            "Train the model on papers fetched for the current query using "
            "**Bayesian Personalized Ranking (BPR)**. This will update "
            "`model_trained.pt`, which is then used in Recommendations."
        )

        if st.button("🧠 Start Training"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                st.info("Training logs will also appear in the terminal.")
                with st.spinner("Optimizing ranking with BPR..."):
                    train_model(query)
                st.success("✅ Training complete. Model saved as `model_trained.pt`.")

    # ------------------------------------------------------ TAB 4 — MODEL DETAILS
    with tabs[3]:
        st.subheader("🧠 Model & System Overview")
        st.markdown(
            """
            **Architecture Highlights**
            - 🔹 SentenceTransformer (`intfloat/e5-base-v2`) for dense text embeddings of paper titles & queries  
            - 🔹 Heterogeneous graph with nodes: papers, authors, venues  
            - 🔹 Meta-paths:
                - `PAP` — Paper → Author → Paper (shared authors)
                - `PVP` — Paper → Venue → Paper (same conference/journal)
            - 🔹 `SemanticAggregator` for multi-hop neighborhood aggregation along each meta-path  
            - 🔹 `SemanticFusion` with attention to weight Identity, PAP, and PVP views  
            - 🔹 Scoring via dot product between fused paper embeddings and projected query embedding  
            - 🔹 Training using **Bayesian Personalized Ranking (BPR)** loss to learn a ranking function  
            """
        )
        st.markdown(
            """
            **Files & Components**
            - `config.py` — hyperparameters, device settings  
            - `data_utils.py` — fetching papers, building meta-path adjacency  
            - `models/aggregator.py` — `SemanticAggregator`, `aggregate_meta_path`  
            - `models/fusion.py` — `SemanticFusion` (attention-based fusion)  
            - `train.py` — BPR training loop, saves `model_trained.pt`  
            - `visualize_graph.py` — NetworkX + PyVis interactive heterogeneous graph  
            - `app.py` — this Streamlit interface (dark cyberpunk theme ✨)  
            """
        )


if __name__ == "__main__":
    main()
