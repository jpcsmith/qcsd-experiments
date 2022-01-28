#!/usr/bin/env python3
import sys
import json
import pathlib
import networkx as nx
import plotly.graph_objects as go
import matplotlib.colors
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def display(filename: str):
    graph_info = json.loads(pathlib.Path(filename).read_text())

    origins = set()
    graph = nx.DiGraph()
    for node in graph_info["nodes"]:
        graph.add_node(node["id"], url=node["url"], origin=node["origin"])
        origins.add(node["origin"])

    for edge in graph_info["links"]:
        graph.add_edge(edge["source"], edge["target"])

    positions = graphviz_layout(graph, prog="dot")
    colour_map = dict(zip(origins, [
        matplotlib.colors.to_hex(c)
        for c in plt.get_cmap("Set2")(list(range(len(origins))))
    ]))

    edge_x = []
    edge_y = []
    for edge in graph.edges():
        pos_x0, pos_y0 = positions[edge[0]]
        pos_x1, pos_y1 = positions[edge[1]]
        edge_x.extend([pos_x0, pos_x1, None])
        edge_y.extend([pos_y0, pos_y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=2, color='#888'),
        hoverinfo='none', mode='lines'
    )

    node_x = []
    node_y = []
    colours = []
    node_text = []
    for (node, data) in graph.nodes(data=True):
        node_x.append(positions[node][0])
        node_y.append(positions[node][1])
        colours.append(colour_map[data["origin"]])
        node_text.append(data["url"])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=colours,
            size=15,
            line_width=2))
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Graph {filename}',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    fig.show()


if __name__ == "__main__":
    display(sys.argv[1])
