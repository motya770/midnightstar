# MidnightStar — Gene-Disease Correlation Discovery Platform

A Streamlit-based discovery platform for non-biology users to search genes/diseases, explore interaction networks visually, and train ML models (GNN, Transformer, VAE) to find novel gene-disease correlation patterns.

## Audience

Researchers who don't have deep biology knowledge. The application must explain all biological concepts in plain language, provide tooltips for technical terms, and present results in an accessible dashboard format.

## Architecture

Multi-page Streamlit app with a shared data and model layer.

```
midnightstar/
├── app.py                    # Streamlit entry point, shared config
├── pages/
│   ├── 1_Search.py           # Gene/disease search with plain-language results
│   ├── 2_Explorer.py         # Interactive network graph exploration
│   ├── 3_Model_Training.py   # Guided ML model configuration and training
│   └── 4_Results.py          # Multi-panel dashboard
├── src/
│   ├── data/
│   │   ├── gwas_client.py    # GWAS Catalog API client
│   │   ├── gtex_client.py    # GTEx API client
│   │   ├── hpa_client.py     # Human Protein Atlas API client
│   │   ├── string_client.py  # STRING database API client
│   │   └── cache.py          # SQLite-backed response caching
│   ├── models/
│   │   ├── gnn.py            # Graph Neural Network (PyTorch Geometric)
│   │   ├── transformer.py    # Transformer model (PyTorch)
│   │   ├── vae.py            # Variational Autoencoder (PyTorch)
│   │   └── trainer.py        # Unified training loop with progress callbacks
│   └── utils/
│       ├── graph_builder.py  # Build NetworkX graphs from API data
│       ├── explainer.py      # Generate plain-language descriptions
│       └── session.py        # Streamlit session state helpers
├── tests/
├── pyproject.toml
└── .streamlit/
    └── config.toml           # Streamlit theme config
```

### Tech Stack

- **Streamlit** — UI framework
- **PyTorch + PyTorch Geometric** — ML models (GNN, Transformer, VAE)
- **NetworkX** — Graph construction and analysis
- **Pyvis** — Interactive network visualization (primary), with Plotly network graphs as fallback. `streamlit-agraph` was considered but is unmaintained (last release Jan 2023). Pyvis generates standalone HTML that Streamlit renders via `st.components.v1.html()`.
- **requests + SQLite** — API data fetching with local caching

## Data Layer

### Common Data Structures

```python
GeneNode(id, ensembl_id, symbol, name, description, organism)
DiseaseNode(id, name, description, category, source)
Association(source_id, target_id, type, score, evidence, data_source)
ExpressionProfile(gene_id, tissue, expression_level, sample_count)
```

Gene symbol-to-Ensembl ID resolution is handled via the MyGene.info API (`mygene` Python package), which provides fast batch lookups. All internal identifiers use Ensembl IDs for cross-source consistency; gene symbols are stored for display.

### API Clients

| Client | Source | Returns | API |
|--------|--------|---------|-----|
| `gwas_client` | GWAS Catalog (EBI) | Gene-disease associations, SNPs, p-values, study metadata | Free REST API |
| `gtex_client` | GTEx Portal | Gene expression across 54 human tissues | Free REST API |
| `hpa_client` | Human Protein Atlas | Protein expression, subcellular location, tissue data | Per-gene JSON endpoints (requires Ensembl ID); bulk TSV download for search/index |
| `string_client` | STRING DB | Protein-protein interaction networks, confidence scores | Free REST API (1 req/sec limit) |

All clients normalize responses into the common data structures above. The rest of the application is source-agnostic.

**HPA access pattern:** Unlike the other three sources, HPA does not provide a search API. The `hpa_client` works in two modes: (1) direct gene lookup via `proteinatlas.org/{ENSEMBL_ID}.json` (requires Ensembl ID from MyGene.info resolution), and (2) a one-time bulk TSV download (`proteinatlas.org/download/proteinatlas.tsv.zip`) that gets indexed into the local SQLite cache for search/auto-suggest. The bulk file (~200MB) is downloaded on first use and refreshed monthly.

### Caching

- SQLite-backed with configurable TTL (default 24 hours)
- Cache key = (source, query, parameters)
- Users can "pin" bulk-downloaded data to prevent expiration
- Force-refresh option available

### Bulk Download

- "Download All Data" button fetches from all 4 APIs in parallel (`concurrent.futures`), with per-source rate limit enforcement (STRING requests throttled to 1/sec independent of other sources)
- Progress bar showing per-source status
- Pinned results persist in cache
- Export merged dataset as CSV/JSON
- For multi-hop queries, STRING requests are batched and queued to respect the rate limit (expected: ~5-20 STRING API calls per bulk download depending on depth)

### Graph Building

- Each API response is converted into NetworkX nodes and edges
- `graph_builder.py` merges data from multiple sources into a unified graph
- Edges carry metadata: source database, confidence score, evidence type

## Page Designs

### 1. Search Page

Entry point. Designed for users who don't know biology.

**Components:**
- Search bar accepting gene names (SP4, HSP60, BRCA1) or disease names (Parkinson's, diabetes)
- Auto-suggest dropdown powered by STRING/GWAS API lookups
- "Download All Data" button for bulk fetching with progress bar

**Results tabs:**
- **Summary** — Plain-language explanation of what the gene does and its disease links
- **Associations** — Table of gene-disease links with source, p-value/confidence, color-coded strength indicator
- **Expression** — Bar chart of tissue activity (GTEx data)
- **Interactions** — Mini network preview of immediate protein partners (STRING), clickable to jump to Explorer

**Plain-language explainer:**
- Auto-generated descriptions aimed at non-biologists
- Example: "SP4 is a protein that helps control which genes are turned on in brain cells. It acts like a switch for neuronal development."
- Built from HPA descriptions + curated templates
- Technical terms get tooltip definitions on hover

### 2. Explorer Page

Visual exploration of gene/disease networks.

**Layout:**
- **Network graph** (~70% of page) using Pyvis (rendered via `st.components.v1.html()`)
  - Nodes colored by type (gene, protein, disease, pathway)
  - Edge thickness proportional to confidence score
  - Click a node to expand its connections
  - Hover shows plain-language tooltip
- **Side panel** (~30%) for selected node details:
  - Name, description, data source
  - Expression sparkline (GTEx)
  - Known disease associations
  - "Train model on this subgraph" button

**Controls:**
- Filter by data source (toggle STRING, GWAS, GTEx, HPA)
- Minimum confidence score slider
- Depth slider (1-3 hops from starting node, default 1)
- Layout algorithm picker (force-directed, circular, hierarchical)
- Reset button

**Navigation:**
- Arrive from Search page or use the inline search input
- Progressive graph building by expanding nodes

### 3. Model Training Page

Guided ML model configuration. No code required.

**Three-column layout:**

**Left — Data Selection:**
- Input: current Explorer subgraph, a saved search, or manual gene selection
- Data summary: node count, edge count, available features
- Feature selector: expression levels, interaction scores, variant associations
- Train/validation/test split slider (default 80/10/10)
- Split strategy: edge-level split for link prediction tasks (remove a percentage of known edges for validation/test). Uses timestamp-based or random edge removal to avoid message-passing leakage where training edges would inform test node embeddings.

**Center — Model Configuration:**
- Model type radio buttons: GNN, Transformer, VAE
- Per-model parameters as sliders/dropdowns:

| Model | Parameters |
|-------|-----------|
| GNN (PyTorch Geometric) | Layers (1-5), hidden dim (32-256), aggregation (mean/max/sum), learning rate |
| Graph Transformer | Heads (1-8), layers (1-6), embedding dim (64-512), RWSE walk length k (8-24), learning rate |
| VAE | Latent dim (8-128), encoder layers (1-4), beta (0.1-10), learning rate |

- Common: epochs (10-500), batch size, early stopping toggle
- "What does this do?" expandable help for each parameter

**Right — Training Monitor:**
- Start/Stop buttons
- Live loss curve (updates each epoch)
- Progress bar with ETA
- Final metrics and "View Results" button

**What each model discovers:**
- **GNN** — Hidden connections in gene interaction networks (genes with similar neighborhoods)
- **VAE** — Latent groupings (genes clustering together may share unknown pathways)
- **Graph Transformer** — Contextual patterns using attention over gene neighborhoods across tissues

## ML Task Formulation

### Prediction Task

The primary task is **link prediction**: given a gene interaction graph with some edges removed, predict which missing edges (gene-gene or gene-disease connections) are most likely to exist. This is how the tool discovers novel correlations.

**Positive examples:** Known edges from the graph (STRING interactions, GWAS associations).
**Negative examples:** Randomly sampled node pairs that have no known connection, at a 1:1 ratio with positives.
**Output:** A score between 0 and 1 representing the predicted probability that two nodes are connected. This is the "Predicted Score" shown in the Results Dashboard.

### Per-Model Input Formulation

- **GNN:** Standard message-passing on the gene interaction graph. Node features = concatenation of normalized expression vector (GTEx, 54 tissues), protein subcellular location one-hot (HPA), and GWAS association count. Edge features = source confidence score.
- **Graph Transformer:** Uses a Graph Transformer architecture (not a sequence Transformer). Input is the same node/edge features as GNN, but attention is computed over k-hop neighborhoods rather than message-passing aggregation. Positional encoding via random-walk structural encoding (RWSE). This captures longer-range dependencies than standard GNN.
- **VAE:** Encodes node feature vectors into a latent space. Reconstruction loss on the adjacency matrix encourages the latent space to capture structural patterns. Genes that cluster together in latent space but lack known edges are surfaced as discovery candidates.

### Evaluation Metrics

- **AUC-ROC** — Primary metric for link prediction quality
- **Average Precision (AP)** — Measures ranking quality of predictions
- **Hits@K (K=10, 50)** — Fraction of true edges ranked in the top K predictions
- For VAE clustering: **Silhouette Score** on the latent embeddings

All metrics are computed on the held-out test edges and displayed in the Training Monitor's "final metrics" section.

### Model Explainability

The "Why?" expandable on the Results Dashboard uses **GNNExplainer** (from PyTorch Geometric) for GNN and Graph Transformer models. GNNExplainer identifies which edges and node features most contributed to a prediction. For VAE, explanations show the nearest neighbors in latent space and their shared features. This is a v1 approach — more sophisticated methods (SHAP, attention visualization) can be added later.

### Training Concurrency

Streamlit runs in a single thread. Model training runs in a separate `multiprocessing.Process` to avoid blocking the UI. Communication between the training process and Streamlit uses a shared `multiprocessing.Queue`:
- The training process pushes epoch metrics (loss, AUC) to the queue each epoch
- The Streamlit UI polls the queue on each rerun (triggered by `st.rerun()` with a 2-second interval during training)
- Stop button sets a shared `multiprocessing.Event` flag that the training loop checks between epochs

### 4. Results Dashboard

Multi-panel discovery view (2x2 grid).

**Top-left — Discovery Network Graph:**
- Same graph engine as Explorer, overlaid with model predictions
- Predicted edges shown as dashed lines in a different color
- Node size scaled by model importance score
- "Show only discoveries" toggle

**Top-right — Summary Panel:**
- Plain-language report of findings
- Top 10 discoveries ranked by confidence
- "Why?" expandable for each finding explaining model reasoning in simple terms

**Bottom-left — Data Table:**
- Sortable, filterable table of all predictions
- Columns: Gene A, Gene B, Predicted Score, Evidence Type, Supporting Sources
- Export to CSV
- Click a row to highlight on the network graph

**Bottom-right — Evidence Panel:**
- Supporting evidence for selected prediction from each data source
- GTEx expression comparison (side-by-side bar charts)
- STRING neighborhood overlap
- GWAS shared disease associations
- Confidence breakdown by data source contribution

**Persistence:**
- Save/load trained models and results to local files
- Compare multiple model runs side-by-side
- Export full report as HTML (PDF export deferred to v2)

## Error Handling

- API failures show inline warnings with retry buttons; the app remains usable with partial data
- Rate limiting (especially STRING's 1 req/sec) handled with automatic backoff
- Model training errors surface in the training monitor with plain-language explanations
- Invalid gene/disease queries show suggestions ("Did you mean...?")

## Testing Strategy

- Unit tests for each API client (mocked responses)
- Unit tests for graph builder (known input -> expected graph structure)
- Unit tests for model forward passes (correct output shapes)
- Integration tests for the data pipeline (search -> fetch -> cache -> graph)
- Manual testing for Streamlit UI interactions

## Compute

- Local-first: models train on the user's CPU/GPU
- Designed so cloud backends (Colab, AWS, Modal) could be added later
- Not built in v1
