# src/models/vae.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj


class GraphVAE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, latent_dim: int = 32,
                 num_layers: int = 2, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.encoder_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.fc_mu = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels, latent_dim)

    def encode(self, x, edge_index):
        h = x
        for layer in self.encoder_layers:
            h = torch.relu(layer(h, edge_index))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.sigmoid(z @ z.t())

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, edge_index):
        adj_pred, mu, logvar = self.forward(x, edge_index)
        adj_true = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0)
        recon_loss = nn.functional.binary_cross_entropy(adj_pred, adj_true)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

    def predict_links(self, x, edge_index, src, dst):
        mu, _ = self.encode(x, edge_index)
        return torch.sigmoid((mu[src] * mu[dst]).sum(dim=-1))

    def get_embeddings(self, x, edge_index):
        mu, _ = self.encode(x, edge_index)
        return mu.detach()
