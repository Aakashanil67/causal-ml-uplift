import networkx as nx

from src.config import OUTCOME_COLS, TREATMENT_COL
from src.dag import build_dag, to_gml


def test_dag_is_acyclic():
    g = build_dag()
    assert nx.is_directed_acyclic_graph(g)


def test_treatment_has_no_parents():
    # the graph's one substantive claim: assignment was randomised, nothing caused it
    g = build_dag()
    assert g.in_degree(TREATMENT_COL) == 0


def test_treatment_points_to_every_outcome():
    g = build_dag()
    for outcome in OUTCOME_COLS:
        assert g.has_edge(TREATMENT_COL, outcome)


def test_every_dag_node_is_a_real_column():
    from src.config import COVARIATE_COLS

    g = build_dag()
    expected = set(COVARIATE_COLS) | {TREATMENT_COL} | set(OUTCOME_COLS)
    assert set(g.nodes) == expected


def test_gml_round_trips_through_networkx():
    g = build_dag()
    gml = to_gml(g)
    parsed = nx.parse_gml(gml)
    assert set(parsed.nodes) == set(g.nodes)
    assert set(parsed.edges) == set(g.edges)
