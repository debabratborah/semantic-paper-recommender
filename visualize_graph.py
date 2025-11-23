from pyvis.network import Network
import networkx as nx


def visualize_graph(works, pap_neighbors, pvp_neighbors):
    """
    Visualizes a heterogeneous graph consisting of:
    - Paper nodes
    - Author nodes
    - Venue nodes
    - Meta-path edges:
        • PAP = Paper–Author–Paper (green)
        • PVP = Paper–Venue–Paper (red)
    """

    G = nx.Graph()

    # =====================================================
    # 1️⃣ Add PAPER nodes
    # =====================================================
    for i, paper in enumerate(works):
        title = paper.get("title", f"Paper {i}")
        G.add_node(
            f"paper_{i}",
            label=title[:60],         # shorten long titles
            color="#6EC6FF",          # light blue
            shape="dot",
            title=f"<b>Paper</b><br>{title}"
        )

    # =====================================================
    # 2️⃣ Add AUTHOR nodes + Paper → Author edges
    # =====================================================
    for i, paper in enumerate(works):
        for author in paper.get("authors", []):
            name = author.get("name")
            if not name:
                continue

            G.add_node(
                f"author_{name}",
                label=name,
                color="#FFA726",      # orange
                shape="diamond",
                title=f"<b>Author</b><br>{name}"
            )

            G.add_edge(
                f"paper_{i}",
                f"author_{name}",
                color="#388E3C",       # dark green
                title="Paper–Author relation"
            )

    # =====================================================
    # 3️⃣ Add VENUE nodes + Paper → Venue edges
    # =====================================================
    for i, paper in enumerate(works):
        venue = paper.get("venue")
        if not venue:
            continue

        G.add_node(
            f"venue_{venue}",
            label=venue,
            color="#AB47BC",          # purple
            shape="box",
            title=f"<b>Venue</b><br>{venue}"
        )

        G.add_edge(
            f"paper_{i}",
            f"venue_{venue}",
            color="#8E24AA",
            title="Paper–Venue relation"
        )

    # =====================================================
    # 4️⃣ META-PATH CONNECTIONS (Paper–Paper)
    # =====================================================

    # -------- PAP: Shared Author ----------
    for i, neighbors in enumerate(pap_neighbors):
        # neighbors may be list or a single id
        if isinstance(neighbors, (list, tuple, set)):
            for n in neighbors:
                if i != n:
                    G.add_edge(
                        f"paper_{i}",
                        f"paper_{n}",
                        color="#43A047",   # green
                        title="PAP: Shared Author"
                    )
        else:
            n = neighbors
            if i != n:
                G.add_edge(
                    f"paper_{i}",
                    f"paper_{n}",
                    color="#43A047",
                    title="PAP: Shared Author"
                )

    # -------- PVP: Shared Venue ----------
    for i, neighbors in enumerate(pvp_neighbors):
        if isinstance(neighbors, (list, tuple, set)):
            for n in neighbors:
                if i != n:
                    G.add_edge(
                        f"paper_{i}",
                        f"paper_{n}",
                        color="#EF5350",   # red
                        title="PVP: Shared Venue"
                    )
        else:
            n = neighbors
            if i != n:
                G.add_edge(
                    f"paper_{i}",
                    f"paper_{n}",
                    color="#EF5350",
                    title="PVP: Shared Venue"
                )

    # =====================================================
    # 5️⃣ PyVis Rendering
    # =====================================================
    net = Network(
        height="850px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="black",
        notebook=False
    )

    # Convert from NetworkX
    net.from_nx(G)

    # Layout — spreads nodes nicely
    net.repulsion(node_distance=220)

    # -------- IMPORTANT FIX ----------
    # Use write_html() instead of show() to avoid errors
    net.write_html("graph.html")

    print("\n[GRAPH] graph.html generated.")
    print("➡️ Open it in your browser to view the interactive graph.\n")
