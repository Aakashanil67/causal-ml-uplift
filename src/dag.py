"""The causal graph: no `dot` binary on this machine, so DoWhy takes a GML string built from a
networkx DiGraph directly, and the figure is drawn with matplotlib instead of graphviz.

Treatment has no parents by construction — this is the graph's one substantive claim, and it is
true here only because `segment` was randomly assigned. `src/confounded.py` builds a second graph
where that assumption is deliberately violated, to show what changes when it doesn't hold.
"""

import matplotlib.pyplot as plt
import networkx as nx

from src.config import COVARIATE_COLS, FIGURES_DIR, OUTCOME_COLS, TREATMENT_COL


def build_dag(confounders: list[str] = COVARIATE_COLS) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(TREATMENT_COL)
    for c in confounders:
        for o in OUTCOME_COLS:
            g.add_edge(c, o)
    for o in OUTCOME_COLS:
        g.add_edge(TREATMENT_COL, o)
    return g


def to_gml(g: nx.DiGraph) -> str:
    """DoWhy's CausalModel(graph=...) parses a GML string, not a file path, for an in-memory
    graph."""
    lines = ["graph [", "directed 1"]
    for node in g.nodes:
        lines.append(f'node [ id "{node}" label "{node}" ]')
    for u, v in g.edges:
        lines.append(f'edge [ source "{u}" target "{v}" ]')
    lines.append("]")
    return "\n".join(lines)


def _layout(g: nx.DiGraph, confounders: list[str]) -> dict:
    pos = {}
    for i, c in enumerate(confounders):
        pos[c] = (0.0, 1.0 - i / max(len(confounders) - 1, 1))
    pos[TREATMENT_COL] = (1.0, 0.5)
    for i, o in enumerate(OUTCOME_COLS):
        pos[o] = (2.0, 1.0 - i / max(len(OUTCOME_COLS) - 1, 1))
    return pos


def draw_dag(g: nx.DiGraph, confounders: list[str], out_path=FIGURES_DIR / "dag.png") -> None:
    pos = _layout(g, confounders)
    fig, ax = plt.subplots(figsize=(11, 7))

    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=confounders,
        node_color="#e8eef7",
        edgecolors="#5b7fb5",
        node_size=2400,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=[TREATMENT_COL],
        node_color="#f6d9a0",
        edgecolors="#b5842b",
        node_size=3200,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=OUTCOME_COLS,
        node_color="#d9ecd9",
        edgecolors="#4a8c4a",
        node_size=2600,
        ax=ax,
    )
    nx.draw_networkx_labels(g, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=[(u, v) for u, v in g.edges if v == TREATMENT_COL or u == TREATMENT_COL],
        edge_color="#b5842b",
        width=1.8,
        arrowsize=18,
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=[(u, v) for u, v in g.edges if v != TREATMENT_COL and u != TREATMENT_COL],
        edge_color="#9aa5b1",
        width=0.7,
        alpha=0.5,
        arrowsize=10,
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )
    ax.set_title(
        "Hillstrom email experiment — causal graph\n"
        "No arrows into treatment: segment assignment was randomised, not chosen",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    g = build_dag()
    print(f"nodes: {g.number_of_nodes()}, edges: {g.number_of_edges()}")
    print(f"treatment has {g.in_degree(TREATMENT_COL)} parents (should be 0)")
    draw_dag(g, COVARIATE_COLS)
    print(f"figure written to {FIGURES_DIR / 'dag.png'}")
    gml = to_gml(g)
    print(f"GML string: {len(gml)} chars")


if __name__ == "__main__":
    main()
