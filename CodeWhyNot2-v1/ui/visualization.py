import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from streamlit_agraph import agraph, Node, Edge, Config

def render_causal_tree(tree_data):
    """
    Render a causal tree visualization from tree_data.
    tree_data: dict with keys 'graph' (nx.DiGraph) and 'intervention_nodes' (list of node indices)
    """
    if tree_data is None or 'graph' not in tree_data or tree_data['graph'] is None:
        st.write("[Causal tree visualization placeholder]")
        return
    G = tree_data['graph']
    intervention_nodes = tree_data.get('intervention_nodes', [])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 3))
    node_colors = ['red' if n in intervention_nodes else 'skyblue' for n in G.nodes]
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color=node_colors, node_size=800, font_size=10, font_weight='bold', edge_color='gray')
    st.pyplot(plt.gcf())
    plt.close()

def render_causal_change_tree(tree):
    """
    Render a causal change tree (prompt/code/score propagation) using NetworkX.
    tree: nx.DiGraph with node attributes: prompt, agent, fidelity, diversity
    """
    if tree is None or len(tree.nodes) == 0:
        st.write("[Causal change tree placeholder]")
        return
    pos = nx.spring_layout(tree)
    plt.figure(figsize=(8, 4))
    labels = {n: f"{tree.nodes[n]['agent']}\n{tree.nodes[n]['prompt'][:18]}...\nF:{tree.nodes[n]['fidelity']} D:{tree.nodes[n]['diversity']}" for n in tree.nodes}
    node_colors = ['#d4edda' if tree.nodes[n]['fidelity'] > 0.8 else ('#fff3cd' if tree.nodes[n]['fidelity'] > 0.5 else '#f8d7da') for n in tree.nodes]
    nx.draw(tree, pos, with_labels=True, labels=labels, node_color=node_colors, node_size=1200, font_size=8, font_weight='bold', edge_color='gray', arrows=True)
    st.pyplot(plt.gcf())
    plt.close()

def render_code_diff(diff_data):
    """Render a code diff viewer from diff_data."""
    st.code(diff_data, language="diff")

def render_scm_graph_editor(graph, key="agraph_default"):
    """
    Render an interactive SCM graph editor using streamlit-agraph.
    """
    # Convert NetworkX graph to agraph nodes/edges
    nodes = [Node(id=str(n), label=str(n)) for n in graph.nodes]
    edges = [Edge(source=str(u), target=str(v)) for u, v in graph.edges]
    config = Config(
        width=700,
        height=400,
        directed=True,
        physics=True,
        hierarchical=False,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', 'renderLabel': False}
    )
    return agraph(nodes=nodes, edges=edges, config=config) 