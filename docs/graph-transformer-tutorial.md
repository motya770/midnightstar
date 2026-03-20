# Graph Transformer for Link Prediction — Step by Step

This tutorial walks through how `GraphTransformerLinkPredictor` works,
from raw graph data to a predicted probability that two nodes are connected.

## The Big Picture

We have a graph (nodes = genes, edges = interactions). We want to predict
**missing edges** — pairs of genes that probably interact but aren't in our
dataset yet. The model does this in three stages:

```
Graph + Node Features
        │
        ▼
┌─────────────────┐
│   1. ENCODE      │  Turn each node into a rich embedding vector
│                  │  that captures its features + position + neighborhood
│   RWSE ──┐      │
│           ├──▶ input_proj ──▶ TransformerConv × 2
│   x ─────┘      │
└────────┬────────┘
         │  z = (num_nodes, 64) embeddings
         ▼
┌─────────────────┐
│   2. DECODE      │  For a candidate edge (u, v), score how likely
│                  │  it is to exist based on their embeddings
│   z[u] · z[v]   │
└────────┬────────┘
         │  scalar logit per edge
         ▼
┌─────────────────┐
│   3. LOSS        │  Compare predictions to ground truth
│   BCE with       │  (positive edges vs negative samples)
│   logits         │
└─────────────────┘
```

---

## Step 1: Positional Encoding (RWSE)

### The problem

Regular transformers process sequences where position matters (word 1, word 2, ...).
Graphs have no natural ordering — node 42 isn't "before" node 43. Without positional
information, the model can't distinguish structurally different nodes that happen to
have similar features.

### The idea: Random Walk Return Probabilities

Imagine you're standing on a node and take a random walk: at each step, you move to
a random neighbor (each neighbor equally likely). **How likely are you to end up back
where you started after k steps?**

This single number tells you a lot about local structure:

- **k=2 (2 steps):** High return probability → your neighbors are densely connected
  to each other (you're in a cluster/clique)
- **k=2, low return:** Your neighbors don't connect back — you might be a bridge
- **k=8 (8 steps):** High return → you're in a tight community that "traps" walks
- **k=8, low return:** You're in a loose, spread-out part of the graph

We compute this for k=1 through k=16, giving each node a 16-dimensional
"structural fingerprint":

```
Node A (hub in tight cluster):  [0.03, 0.15, 0.12, 0.10, 0.09, ...]  ← high return probs
Node B (bridge between groups):  [0.01, 0.02, 0.01, 0.02, 0.01, ...]  ← low return probs
```

### How it's computed: the Hutchinson estimator

Computing the exact return probability for every node is expensive (you'd need
N separate random walks). Instead, we use a trick from numerical linear algebra.

The return probability at step k for node i is the diagonal entry `(RW^k)[i,i]`
of the random walk matrix raised to the k-th power. Computing full matrix powers
is O(N^3) — way too expensive for 70K nodes.

**Hutchinson's trace estimator** says: if you pick random vectors z with entries
+1 or -1, then:

```
E[ z[i] * (M @ z)[i] ] = M[i,i]
```

In plain English: multiply M by a random vector, then element-wise multiply with
the original random vector, and the expected value at each position gives you the
diagonal of M.

In code:

```python
# Create 32 random ±1 vectors (probes), each of length num_nodes
probes_orig = torch.sign(torch.randn(num_nodes, 32))
probes = probes_orig.clone()

for k in range(16):  # walk lengths 1..16
    # One step of random walk: probes = RW @ probes
    # Done via scatter (sparse matrix-vector multiply)
    src_vals = probes[row] * (1.0 / degree[row])  # weight by 1/degree
    new_probes = zeros_like(probes)
    new_probes.scatter_add_(0, col, src_vals)      # accumulate at destinations
    probes = new_probes

    # Hutchinson estimate of diag(RW^k)
    rw_diag[:, k] = (probes_orig * probes).mean(dim=1)
```

Each loop iteration is one random walk step applied to all 32 probe vectors
simultaneously. After k steps, `probes` contains `RW^k @ probes_orig`, and
the element-wise product with `probes_orig` estimates the diagonal.

32 probes gives a reasonable approximation. More probes = more accurate but slower.

### The trainable projection

The raw 16-dim walk probabilities are then passed through a learned linear layer:

```python
pe = self.linear(rw_diag)  # (num_nodes, 16) → (num_nodes, rwse_dim)
```

This lets the model learn **which walk lengths matter** for link prediction.
Maybe 2-step return probability is critical but 14-step doesn't help — the
linear layer can learn to upweight the useful ones and ignore the rest.

### Caching

The walk probabilities only depend on graph structure, which doesn't change
during training. So we compute them once before training starts and reuse the
cached values every epoch. Only the linear projection (which is tiny) runs
each forward pass.

---

## Step 2: Input Projection

Each node now has two pieces of information:
- `data.x` — its original features (e.g., gene expression values), shape `(N, in_channels)`
- `pe` — its positional encoding from RWSE, shape `(N, rwse_dim)`

We concatenate them and project to a uniform hidden dimension:

```python
x = torch.cat([data.x, pe], dim=-1)   # (N, in_channels + rwse_dim)
x = self.input_proj(x)                 # (N, hidden_channels)  e.g. (N, 64)
```

This is like the embedding layer in a language transformer — it puts everything
into the same dimensional space before the transformer layers process it.

---

## Step 3: TransformerConv Layers

This is where the actual "graph transformer" happens. Each layer lets every node
attend to its neighbors and update its representation.

### What TransformerConv does

For each node i, it computes attention-weighted messages from its neighbors:

```
For each neighbor j of node i:
    Q = W_q @ x[i]          # query: "what am I looking for?"
    K = W_k @ x[j]          # key: "what do I offer?"
    V = W_v @ x[j]          # value: "here's my information"

    attention = softmax(Q · K / sqrt(d))   # how relevant is neighbor j?
    message_j = attention * V               # weighted information from j

x_new[i] = sum of all message_j            # aggregate neighbor messages
```

This is standard multi-head attention, but restricted to graph neighbors
(not all-pairs like in language transformers). With 4 heads and 64 hidden channels,
each head works in 16 dimensions (64/4), then the heads are concatenated back to 64.

### Residual connection + LayerNorm

```python
x = norm(x + layer(x, data.edge_index))   # residual + normalize
x = torch.relu(x)                          # non-linearity
```

- **Residual connection** (`x + layer(...)`) — the output is the old embedding
  plus the new information from neighbors. This prevents the "washing out" problem
  where deep networks forget the original node features.
- **LayerNorm** — stabilizes training by normalizing the embedding magnitudes.
- **ReLU** — introduces non-linearity so the model can learn complex patterns.

### Two layers = 2-hop neighborhood

Layer 1: each node collects info from direct neighbors (1 hop).
Layer 2: each node collects info from its neighbors' neighbors (2 hops),
because those neighbors already incorporated their own neighbors in layer 1.

After 2 layers, each node's embedding reflects a 2-hop neighborhood
around it, weighted by attention.

---

##  Step 4: Decode — Predicting Edges

Given embeddings for all nodes, we score a candidate edge (u, v):

```python
def decode_logits(self, z, src, dst):
    return (z[src] * z[dst]).sum(dim=-1)
```

This is a **dot product** between the two node embeddings:

```
z[u] = [0.3, -0.1, 0.8, ...]   # 64-dim embedding of node u
z[v] = [0.2,  0.5, 0.7, ...]   # 64-dim embedding of node v

logit = 0.3*0.2 + (-0.1)*0.5 + 0.8*0.7 + ...
      = scalar value (higher = more likely connected)
```

**Intuition:** if two nodes have similar embeddings (point in the same direction
in 64-dim space), their dot product is large → model predicts they're connected.
If their embeddings point in different directions, dot product is small/negative
→ model predicts no edge.

For inference (actual predictions), we apply sigmoid to get a probability:

```python
def decode(self, z, src, dst):
    return torch.sigmoid(self.decode_logits(z, src, dst))  # → [0, 1]
```

For training, we use raw logits with `binary_cross_entropy_with_logits` which
is numerically stable (avoids log(0) when sigmoid saturates).

---

## Step 5: Training — What the Loss Function Sees

Each training step:

1. **Positive edges:** Real edges held out from message passing (the model can't
   see them in the graph — it has to predict them from structure alone)
2. **Negative edges:** Randomly sampled non-edges (pairs of nodes with no known connection)
3. **Forward pass:** Encode all nodes → decode each candidate edge → get logits
4. **Loss:** Binary cross-entropy between predictions and labels (1 for real edges,
   0 for fake edges)

```
Positive edge (Gene_A, Gene_B):  logit = 3.2  → sigmoid = 0.96  ✓ (label = 1)
Negative edge (Gene_A, Gene_X):  logit = -1.5 → sigmoid = 0.18  ✓ (label = 0)
Positive edge (Gene_C, Gene_D):  logit = -0.3 → sigmoid = 0.43  ✗ (label = 1, wrong!)
```

The loss penalizes wrong predictions, and backpropagation updates all trainable
parameters: TransformerConv weights, input projection, and the RWSE linear layer.

### Why supervision edges are excluded from message passing

This is critical. If the edge (A, B) is both in the message-passing graph AND
a positive supervision edge, the model can trivially learn "if I can send a message
along this edge, it exists." That doesn't generalize — at test time, the edges
we want to predict are NOT in the graph.

So we split training edges into two groups:
- **85% → message passing:** the graph structure the model can see
- **15% → supervision:** edges the model must predict without seeing them

---

## Putting It All Together

```
Input: graph with 70K nodes, 1.9M edges, node features

BEFORE TRAINING (once):
  Precompute RWSE cache: random walk return probs for all nodes  → ~4 seconds

EACH EPOCH (~2 seconds with cache):
  1. RWSE linear projection: cached walk probs → 16-dim positional encoding
  2. Concatenate features + PE → project to 64-dim
  3. TransformerConv layer 0: attend to neighbors, update embeddings
  4. TransformerConv layer 1: attend to neighbors again (now 2-hop info)
  5. For each supervision edge and negative sample:
     - Dot product of source and destination embeddings → logit
     - BCE loss against ground truth
  6. Backpropagate → update all weights
  7. Evaluate on held-out validation edges → AUC score

INFERENCE:
  Same encode pipeline → decode any (u, v) pair → sigmoid → probability
```

## Hyperparameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `hidden_channels` | 64 | Embedding dimension — larger = more expressive but slower |
| `num_layers` | 2 | Transformer depth — more layers = larger receptive field |
| `num_heads` | 4 | Attention heads — each head learns different relationship types |
| `rwse_dim` | 16 | Size of positional encoding after projection |
| `rwse_walk_length` | 16 | Max random walk steps — longer captures more global structure |
| `lr` | 0.001 | Learning rate |
| `edge_dropout` | 0.3 | Fraction of edges randomly dropped during mini-batch encode |
