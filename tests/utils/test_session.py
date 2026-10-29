# tests/utils/test_session.py
import networkx as nx

from src.utils import session


def test_set_graph_issues_a_new_token_per_graph(monkeypatch):
    monkeypatch.setattr(session.st, "session_state", {})
    session.set_graph(nx.Graph(), name="a")
    first = session.get_graph_token()
    session.set_graph(nx.Graph(), name="b")
    second = session.get_graph_token()
    assert first and second and first != second
