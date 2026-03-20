# src/models/vae.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling


class GraphVAE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, latent_dim: int = 32,
                 num_layers: int = 2, beta: float = 1.0,
                 gwas_vocab_size: int = 0, gwas_num_categories: int = 0, gwas_embed_dim: int = 32):
        super().__init__()
        self.beta = beta
        from .gnn import GWASEncoder
        self.gwas_encoder = None
        total_in = in_channels
        if gwas_vocab_size > 1:
            self.gwas_encoder = GWASEncoder(gwas_vocab_size, gwas_num_categories, gwas_embed_dim)
            total_in = in_channels + gwas_embed_dim
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(SAGEConv(total_in, hidden_channels))
        for _ in range(num_layers - 1):
            self.encoder_layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.fc_mu = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels, latent_dim)

    def _get_x(self, data):
        x = data.x
        if self.gwas_encoder is not None and hasattr(data, "gwas_token_ids"):
            gwas_feat = self.gwas_encoder(data.gwas_token_ids, data.gwas_scores, data.gwas_cat_ids)
            x = torch.cat([x, gwas_feat], dim=-1)
        return x

    def _encode_raw(self, x, edge_index):
        h = x
        for layer in self.encoder_layers:
            h = torch.relu(layer(h, edge_index))
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, data):
        """Encode returning z (reparameterized). Stores mu/logvar for KL loss."""
        x = self._get_x(data)
        mu, logvar = self._encode_raw(x, data.edge_index)
        self._mu = mu
        self._logvar = logvar
        return self.reparameterize(mu, logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, src, dst):
        """Edge-level decode — returns probabilities for given edges."""
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

    def decode_logits(self, z, src, dst):
        """Edge-level decode — returns raw logits for given edges."""
        return (z[src] * z[dst]).sum(dim=-1)

    def kl_loss(self):
        """KL divergence from the most recent encode call."""
        return -0.5 * torch.mean(1 + self._logvar - self._mu.pow(2) - self._logvar.exp())

    def predict_links(self, x, edge_index, src, dst, data=None):
        inp = self._get_x(data) if data is not None else x
        mu, _ = self._encode_raw(inp, edge_index)
        return torch.sigmoid((mu[src] * mu[dst]).sum(dim=-1))

    def get_embeddings(self, x, edge_index, data=None):
        inp = self._get_x(data) if data is not None else x
        mu, _ = self._encode_raw(inp, edge_index)
        return mu.detach()
