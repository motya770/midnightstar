import streamlit as st
import networkx as nx


def get_graph() -> nx.Graph | None:
    return st.session_state.get("graph")


def set_graph(graph: nx.Graph):
    st.session_state["graph"] = graph


def get_selected_node() -> str | None:
    return st.session_state.get("selected_node")


def set_selected_node(node_id: str):
    st.session_state["selected_node"] = node_id


def get_search_query() -> str:
    return st.session_state.get("search_query", "")


def set_search_query(query: str):
    st.session_state["search_query"] = query


def get_training_results() -> dict | None:
    return st.session_state.get("training_results")


def set_training_results(results: dict):
    st.session_state["training_results"] = results


def get_model_predictions() -> list[dict] | None:
    return st.session_state.get("model_predictions")


def set_model_predictions(predictions: list[dict]):
    st.session_state["model_predictions"] = predictions
