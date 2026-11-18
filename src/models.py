"""PNA for molecular property prediction.

PNA (Corso et al., NeurIPS 2020) combines several aggregators with
degree-scalers. The layer itself comes from PyTorch Geometric; this file wires
it to OGB's atom/bond encoders and a graph-level head, and keeps the same
shape of model I used for transaction graphs in AML-GNN.
"""
import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d, Linear, ModuleList, Sequential, ReLU
from torch_geometric.nn import PNAConv, global_mean_pool

AGGREGATORS = ["mean", "max", "min", "std"]
SCALERS = ["identity", "amplification", "attenuation"]


class PNANet(torch.nn.Module):
    def __init__(self, deg, emb_dim=80, num_layers=4, dropout=0.3, towers=4, num_tasks=1):
        super().__init__()
        self.dropout = dropout
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoders = ModuleList()
        self.convs = ModuleList()
        self.norms = ModuleList()
        for _ in range(num_layers):
            self.bond_encoders.append(BondEncoder(emb_dim))
            self.convs.append(
                PNAConv(
                    in_channels=emb_dim,
                    out_channels=emb_dim,
                    aggregators=AGGREGATORS,
                    scalers=SCALERS,
                    deg=deg,
                    edge_dim=emb_dim,
                    towers=towers,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )
            )
            self.norms.append(BatchNorm1d(emb_dim))

        self.head = Sequential(
            Linear(emb_dim, emb_dim // 2), ReLU(), Linear(emb_dim // 2, num_tasks)
        )

    def forward(self, data):
        x = self.atom_encoder(data.x)
        for conv, norm, bond_encoder in zip(self.convs, self.norms, self.bond_encoders):
            edge_attr = bond_encoder(data.edge_attr)
            h = conv(x, data.edge_index, edge_attr)
            h = F.relu(norm(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # residual
        return self.head(global_mean_pool(x, data.batch))
