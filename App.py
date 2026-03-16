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
    page_title="Academic GNN Paper Recommender",
    page_icon="📚",
    layout="wide"
)


# ===========================================================
# BACKGROUND + CYBER GRAPH CSS
# ===========================================================
def inject_cyberpunk_css():
    st.markdown(
        """
    <style>

    /* === BACKGROUND === */
    .stApp {
        background: linear-gradient(160deg, #090a10 10%, #0b1121 60%, #050816 100%) !important;
        background-attachment: fixed;
        background-size: cover;
        color: #E6EBFF !important;
    }

    body, .block-container {
        background: transparent !important;
    }

    /* GRAPH NETWORK VIBES OVERLAY */
    body::before {
        content: "";
        position: fixed;
        width: 100%;
        height: 100%;
        top:0;
        left:0;
        opacity:0.08;
        pointer-events:none;
        background-image:
            url("https://images.unsplash.com/photo-1535223289827-42f1e9919769?auto=format&fit=crop&w=1400&q=60"),
            url("https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&w=1400&q=60");
        background-size: cover;
        background-position: center;
        mix-blend-mode: overlay;
        filter: blur(4px) brightness(1.2);
    }

    /* === HEADER === */
    .neon-title {
        text-align:center;
        font-size:3.0rem;
        font-weight:900;
        text-transform:capitalize;
        margin-top:0.5rem;
        margin-bottom:1.4rem;
        background: linear-gradient(90deg,#00f7ff,#2bb3ff,#ff2bfd,#f1ff00);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        letter-spacing:0.6px;
        text-shadow:0 0 35px rgba(0,255,255,.25);
    }

    /* === SIDEBAR CLEAN === */
    [data-testid="stSidebar"] {
        background: rgba(7, 10, 25, .65);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0,255,240,0.15);
    }
    [data-testid="stSidebar"] * {
        color: #E6EBFF !important;
    }

    /* === RECOMMENDATION CARDS === */
    .rec-card {
        background: rgba(10,14,32,0.82);
        border-radius: 16px;
        padding: 16px 18px;
        border: 1px solid rgba(0,255,240,0.18);
        box-shadow:0 0 18px rgba(0,0,0,0.65);
        margin-bottom: 14px;
        transition: all .22s ease-out;
    }
    .rec-card:hover {
        transform: translateY(-3px) scale(1.01);
        border-color: rgba(255,0,200,.45);
        box-shadow:0 0 28px rgba(0,255,240,.25);
    }

    .rec-title { font-size:1.05rem; font-weight:700; color:#E8EDFF; }
    .rec-meta  { font-size:0.9rem; color:#A4A8D0; }

    .score-tag {
        padding:2px 8px;
        border-radius:999px;
        background:rgba(255,0,200,0.12);
        border:1px solid rgba(255,0,200,0.55);
        color:#FFD6FA;
    }
    .rec-link a {
        color:#48d9ff !important;
        text-decoration:none;
        font-size:0.92rem;
    }

    </style>
    """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("intfloat/e5-base-v2")


# ===========================================================
# RECOMMEND ENGINE
# ===========================================================
def get_recommendations(query, limit, top_k):
    device = params["device"]
    model = load_sentence_model()

    works = fetch_papers(query, limit=limit)
    if not works:
        return [], None, False, works

    titles = [w.get("title", "") for w in works]

    paper_emb = torch.tensor(
        model.encode(titles),
        dtype=torch.float,
        device=device,
    )

    pap_neighbors, pvp_neighbors = build_meta_path_adjs(works)

    # Load modules
    ag_pap = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pvp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    proj_identity = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * params["L"]).to(device)
    fusion = SemanticFusion(params["semantic_proj_dim"] * params["L"], params["semantic_proj_dim"]).to(device)
    proj_user = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * params["L"]).to(device)

    trained = False
    if os.path.exists("model_trained.pt"):
        ckpt = torch.load("model_trained.pt", map_location=device)
        ag_pap.load_state_dict(ckpt["aggregator_pap"])
        ag_pvp.load_state_dict(ckpt["aggregator_pvp"])
        proj_identity.load_state_dict(ckpt["proj_identity"])
        fusion.load_state_dict(ckpt["fusion"])
        proj_user.load_state_dict(ckpt["proj_user"])
        trained = True

    E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, ag_pap, L=params["L"])
    E_pvp = aggregate_meta_path(works, paper_emb, pvp_neighbors, ag_pvp, L=params["L"])
    E_identity = torch.relu(proj_identity(paper_emb))

    fused_items, meta_weights = fusion([E_identity, E_pap, E_pvp])
    fused_items = F.normalize(fused_items, dim=1)

    q_emb = torch.tensor(model.encode([query])[0], dtype=torch.float, device=device)
    q_proj = F.normalize(proj_user(q_emb), dim=0)

    scores = torch.mv(fused_items, q_proj)
    k = min(top_k, len(scores))
    topk = torch.topk(scores, k=k)

    recs = []
    for rank, (idx, score) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
        p = works[idx]
        recs.append({
            "rank": rank,
            "title": p.get("title", "N/A"),
            "authors": [a.get("name") for a in p.get("authors", [])][:5],
            "venue": p.get("venue", "N/A"),
            "year": p.get("year", "N/A"),
            "url": p.get("url", "N/A"),
            "score": float(score),
            "index": idx,
        })

    return recs, meta_weights, trained, works


# ===========================================================
# UI
# ===========================================================
def main():
    inject_cyberpunk_css()

    st.markdown(
        """
        <div class='neon-title'>
        Heterogeneous Graph Neural Network For Academic Paper Recommendation
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 🧭 Sidebar controls — CLEAN
    query = st.text_input("🔍 Research Topic", "Neural Networks")
    limit = st.sidebar.slider("Paper Fetch Count", 10, 50, 25)
    top_k = st.sidebar.slider("Top-K Results", 3, 20, 10)

    tabs = st.tabs(["📚 Recommendations", "🌐 Graph View", "🧠 Train Model"])

    # ------------------------------------------------ TAB 1
    with tabs[0]:
        if st.button("🚀 Recommend"):
            with st.spinner("Running semantic GNN..."):
                recs, meta_weights, trained, works = get_recommendations(query, limit, top_k)

            if not recs:
                st.warning("No papers found.")
            else:
                for r in recs:
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-title">{r['rank']}. {r['title']}</div>
                        <div class="rec-meta"><b>Authors:</b> {", ".join(r['authors'])}</div>
                        <div class="rec-meta"><b>Venue:</b> {r['venue']} | <b>Year:</b> {r['year']}</div>
                        <div style="margin-top:6px;">
                            <span class="score-tag">Score: {r['score']:.4f}</span>
                        </div>
                        <div class="rec-link" style="margin-top:6px;">
                            {'<a href="'+r['url']+'" target="_blank">🔗 Open Paper</a>' if r['url']!='N/A' else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ------------------------------------------------ TAB 2
    with tabs[1]:
        st.write("Visualize paper graph relationships.")
        if st.button("🌐 Build Graph"):
            works = fetch_papers(query, limit)
            pap, pvp = build_meta_path_adjs(works)
            visualize_graph(works, pap, pvp)

            if os.path.exists("graph.html"):
                with open("graph.html") as f:
                    html(f.read(), height=780, scrolling=True)

    # ------------------------------------------------ TAB 3
    with tabs[2]:
        st.write("Train model using BPR loss")
        if st.button("🏋️ Train"):
            with st.spinner("Training..."):
                train_model(query)
            st.success("Training Done.")


if __name__ == "__main__":
    main()
