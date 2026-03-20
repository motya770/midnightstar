# CLAUDE.md — MidnightStar

## Project Overview

MidnightStar is a gene-disease correlation discovery platform for non-biology researchers. It provides a Streamlit-based UI to search genes/diseases, explore protein-interaction networks, and train ML models (GNN, Graph Transformer, VAE) to discover novel gene-disease correlations.

**Tech stack:** Python 3.11+, Streamlit, PyTorch, PyTorch Geometric, NetworkX, SQLite

## Repository Structure

```
app.py                        # Streamlit entry point
pages/                        # Streamlit multi-page views (0_Download → 9_Structure_Viewer)
src/
  data/                       # API clients, caching, data models
    models.py                 # Frozen dataclasses: GeneNode, DiseaseNode, Association, ExpressionProfile
    cache.py                  # SQLite cache with TTL (24h) + pinning
    gene_resolver.py          # MyGene.info symbol→Ensembl ID resolution
    gwas_client.py            # GWAS Catalog (EBI) API
    gtex_client.py            # GTEx Portal expression API
    hpa_client.py             # Human Protein Atlas API
    string_client.py          # STRING DB protein interactions (rate-limited 1 req/sec)
    alphafold_client.py       # AlphaFold structure integration
    bulk_downloader.py        # Parallel multi-source fetching
  models/                     # ML models and training
    gnn.py                    # Graph Neural Network (SAGEConv)
    graph_transformer.py      # Transformer with RWSE (Random Walk Structural Encoding)
    vae.py                    # Variational Autoencoder for graphs
    trainer.py                # Unified training loop (CPU/MPS/CUDA)
    remote_trainer.py         # Modal-based distributed training
    modal_app.py              # Modal serverless functions
  utils/
    graph_builder.py          # NetworkX graph construction from API data
    text_explainer.py         # Plain-language descriptions for non-biologists
    session.py                # Streamlit session state management
tests/                        # pytest suite, mirrors src/ structure
docs/                         # Design specs, implementation plans, articles
```

## Common Commands

```bash
# Run the app
streamlit run app.py

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src

# Install dependencies
pip install -e ".[dev]"

# Install with GPU support
pip install -e ".[dev,gpu]"
```

## Code Conventions

- **Data models:** Immutable frozen dataclasses with `to_dict()` methods
- **Type hints:** Python 3.11+ syntax throughout (`str | None`, not `Optional[str]`)
- **Naming:** snake_case for functions/variables, CamelCase for classes
- **API clients:** Return normalized data structures (Association, GeneNode, ExpressionProfile); use logging for error fallbacks
- **ML models:** Inherit `nn.Module`, separate `encode()`/`decode()` methods, support multi-device (cpu/mps/cuda)
- **Caching:** SQLite-backed, keys are SHA256(source + query + params), 24h TTL default
- **Testing:** Use `responses` library to mock HTTP requests; test files mirror src/ layout
- **Streamlit pages:** Module-level client imports, sidebar cache controls, multi-column layouts

## Architecture Notes

- API clients in `src/data/` each wrap a specific bioinformatics API with rate limiting and error handling
- The graph builder (`src/utils/graph_builder.py`) constructs NetworkX graphs from combined API data
- Three ML model architectures are available: GNN (SAGEConv), Graph Transformer (with RWSE), and VAE
- The trainer supports local training on CPU/MPS/CUDA and remote training via Modal
- SQLite cache enables offline-capable operation with per-source clearing

## Dependencies

Defined in `pyproject.toml`. Key libraries: streamlit (>=1.40), torch (>=2.2), torch-geometric (>=2.5), networkx (>=3.2), mygene (>=3.2.2), pyvis (>=0.3.2), plotly (>=5.18), pandas (>=2.1), scikit-learn (>=1.4), requests (>=2.31).

Dev extras: pytest (>=8.0), pytest-cov, responses. GPU extras: modal (>=0.67).
