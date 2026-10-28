import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from model_gbn import DAG_EDGES

graph = nx.DiGraph(DAG_EDGES)

"""
ensure to manually position sparsely otherise you'll get 
a flattened Linked list possibly hiding edges
"""
positions = {
    "Yield_10Y": (0, 0),
    "XLK_Return": (2, 0),
    "VIX_Return": (4, -1.5),
    "Asset_1_Return": (6, -1.5),
    "Asset_2_Return": (8, 0),
}
labels = {
    "Yield_10Y": "10Y yield",
    "XLK_Return": "XLK\nreturn",
    "VIX_Return": "VIX\nreturn",
    "Asset_1_Return": "Asset 1\nreturn",
    "Asset_2_Return": "Asset 2\nreturn",
}

fig, ax = plt.subplots(figsize=(12, 5))
nx.draw_networkx_nodes(graph, positions, node_size=4800, node_color="lightblue", ax=ax)
nx.draw_networkx_edges(
    graph, positions,
    arrows=True,
    arrowsize=20,
    node_size=4800,
    width=2,
    ax=ax,
)
nx.draw_networkx_labels(graph, positions, labels=labels, font_size=10, ax=ax)
ax.set_xlim(-0.8, 8.8)
ax.set_ylim(-2.3, 0.8)
ax.axis("off")
fig.tight_layout()

output_path = Path(__file__).with_name("bn_structure.png")
fig.savefig(output_path, dpi=160, bbox_inches="tight")
print(f"Saved BN diagram to {output_path}")
plt.show()
