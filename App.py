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


# =====================================================================================
# STREAMLIT PAGE CONFIG
# =====================================================================================
st.set_page_config(
    page_title="Heterogeneous GNN Paper Recommender",
    page_icon="📚",
    layout="centered"
)


# =====================================================================================
# BACKGROUND UI (Graph Animation + Research Images)
# =====================================================================================
html("""
<style>
body, .stApp {
    background: #01010C;
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}

/* Paper images behind graph */
.bg-layer {
    background-image: 
        url('https://images.unsplash.com/photo-1537432376769-00a2b8e1a87b'),
        url('https://images.unsplash.com/photo-1529333166437-7750a6dd5a70');
    background-size: 60%, 45%;
    background-repeat: no-repeat;
    background-position: left -150px top 120px, right -160px bottom 80px;
    opacity: 0.06;
    width: 100vw;
    height: 100vh;
    position: fixed;
    z-index: -3;
}

/* Fullscreen canvas for graph */
#bgcanvas {
    position: fixed;
    top:0; left:0;
    width:100vw;
    height:100vh;
    z-index:-2;
}
</style>

<div class="bg-layer"></div>
<canvas id="bgcanvas"></canvas>

<script>
let canvas = document.getElementById('bgcanvas');
let ctx = canvas.getContext('2d');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const N = 60;
const particles = [];

for(let i=0;i<N;i++){
    particles.push({
        x:Math.random()*canvas.width,
        y:Math.random()*canvas.height,
        dx:(Math.random()-0.5)*0.5,
        dy:(Math.random()-0.5)*0.5
    });
}

function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle="rgba(5,5,15,0.90)";
    ctx.fillRect(0,0,canvas.width,canvas.height);

    ctx.strokeStyle="rgba(0,255,255,0.13)";
    ctx.lineWidth=0.6;

    // connections
    for(let i=0;i<N;i++){
        let a = particles[i];
        for(let j=i+1;j<N;j++){
            let b = particles[j];
            let d = Math.hypot(a.x-b.x, a.y-b.y);

            if(d < 190){
                ctx.globalAlpha = 1 - d/190;
                ctx.beginPath();
                ctx.moveTo(a.x,a.y);
                ctx.lineTo(b.x,b.y);
                ctx.stroke();
            }
        }
    }

    // nodes
    for(let p of particles){
        p.x+=p.dx; p.y+=p.dy;
        if(p.x < 0 || p.x > canvas.width) p.dx *= -1;
        if(p.y < 0 || p.y > canvas.height) p.dy *= -1;

        ctx.beginPath();
        ctx.fillStyle="rgba(0,255,255,0.85)";
        ctx.arc(p.x,p.y,2.3,0,2*Math.PI);
        ctx.fill();
    }
}
setInterval(draw, 45);
</script>
""", height=0)


# =====================================================================================
# MODEL CACHE
# =====================================================================================
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("intfloat/e5-base-v2")


# =====================================================================================
# RECOMMENDATION PIPELINE
# =====================================================================================
def recommend(query, limit, top_k):
    device = params["device"]
    model = load_sentence_model()

    works = fetch_papers(query, limit=limit)
    if not works:
        return [], None, False, None

    titles = [w.get("title", "") for w in works]

    paper_emb = torch.tensor(
        model.encode(titles),
        dtype=torch.float,
        device=device
    )

    pap_neighbors, pvp_neighbors = build_meta_path_adjs(works)

    agg_pap = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    agg_pvp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    proj_id = nn.Linear(params["st_dim"], params["semantic_proj_dim"] * params["L"]).to(device)
    fusion = SemanticFusion(params["semantic_proj_dim"]*params["L"], params["semantic_proj_dim"]).to(device)
    proj_user = nn.Linear(params["st_dim"], params["semantic_proj_dim"]*params["L"]).to(device)

    trained = False
    if os.path.exists("model_trained.pt"):
        ckpt = torch.load("model_trained.pt", map_location=device)
        agg_pap.load_state_dict(ckpt["aggregator_pap"])
        agg_pvp.load_state_dict(ckpt["aggregator_pvp"])
        proj_id.load_state_dict(ckpt["proj_identity"])
        fusion.load_state_dict(ckpt["fusion"])
        proj_user.load_state_dict(ckpt["proj_user"])
        trained = True

    E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, agg_pap, L=params["L"])
    E_pvp = aggregate_meta_path(works, paper_emb, pvp_neighbors, agg_pvp, L=params["L"])
    E_id = F.relu(proj_id(paper_emb))

    fused_items, meta_weights = fusion([E_id, E_pap, E_pvp])
    fused_items = F.normalize(fused_items, dim=1)

    q_emb = model.encode([query])[0]
    q_emb = torch.tensor(q_emb, dtype=torch.float, device=device)
    q_proj = F.normalize(proj_user(q_emb), dim=0)

    scores = torch.mv(fused_items, q_proj)
    k = min(top_k, scores.size(0))

    topk = torch.topk(scores, k=k)

    out = []
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        p = works[idx]
        out.append({
            "index": idx,
            "score": float(score),
            "title": p.get("title"),
            "authors": [a.get("name") for a in p.get("authors", [])][:4],
            "venue": p.get("venue"),
            "year": p.get("year"),
            "url": p.get("url")
        })
    return out, meta_weights, trained, works


# =====================================================================================
# MAIN UI
# =====================================================================================
def main():

    # TITLE
    st.markdown(
        """
        <h1 style='text-align:center;
                  font-size:45px;
                  background:linear-gradient(90deg,#00fff7,#ff00d2,#ffd05e);
                  -webkit-background-clip:text;
                  color:transparent;
                  font-weight:900'>
            Heterogeneous Graph Neural Network For<br>
            Academic Paper Recommendation
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    # INPUT
    query = st.text_input(
        "🔎 Research Topic",
        placeholder="e.g., Graph Neural Networks"
    )
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Papers to Fetch", 10, 50, 25)
    with col2:
        top_k = st.slider("Top K Results", 5, 20, 10)

    tabs = st.tabs(["📚 Recommendations", "🌐 Graph View", "🧠 Train Model"])

    # =================================================== Tab 1 Recommendation
    with tabs[0]:
        if st.button("🚀 Recommend"):
            with st.spinner("Running Meta-path GNN..."):
                recs, weights, trained, works = recommend(query, limit, top_k)

            if not recs:
                st.error("No papers found!")
            else:
                if trained:
                    st.success("Model trained ✔")
                else:
                    st.warning("Using untrained weights ⚠")

                for r in recs:
                    st.markdown(
                        f"""
                        <div style="
                            background:rgba(255,255,255,0.04);
                            padding:14px;border-radius:10px;margin:7px 0;
                            border:1px solid rgba(0,255,255,0.18);">
                        <b>{r['title']}</b><br>
                        <span style='color:#8AF6FF'>Authors:</span> {", ".join(r['authors'])}<br>
                        <span style='color:#FF9DDD'>Score:</span> {r['score']:.4f}<br>
                        <a href="{r['url']}" target="_blank">🔗 Open Paper</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # =================================================== Tab 2 Graph View
    with tabs[1]:
        if st.button("🌐 Visualize Graph"):
            works = fetch_papers(query, limit)
            pap, pvp = build_meta_path_adjs(works)
            visualize_graph(works, pap, pvp)

            with open("graph.html","r") as f:
                html(f.read(), height=700)

    # =================================================== Tab 3 Train
    with tabs[2]:
        if st.button("🧠 Train on Topic"):
            st.info("Training… Check console logs.")
            train_model(query)
            st.success("Model trained and saved ✔")


if __name__ == "__main__":
    main()
