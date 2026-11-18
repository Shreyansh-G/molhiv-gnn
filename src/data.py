"""Dataset loading, scaffold split and degree histogram for ogbg-molhiv.

Thin wrapper over the official OGB loaders (MIT, snap-stanford/ogb) so every
model in this repo trains and is scored on exactly the same split.
"""
import torch
import torch.serialization
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

DATASET = "ogbg-molhiv"


def _allow_pyg_globals():
    """Let ogb's `torch.load` read its own processed cache on torch >= 2.6.

    torch 2.6 flipped the default of `torch.load(weights_only=...)` from False
    to True. ogb 1.3.6 calls `torch.load(self.processed_paths[0])` with no
    arguments, so loading ogbg-molhiv dies with an UnpicklingError complaining
    about `torch_geometric.data.data.DataEdgeAttr`. Rather than switch the load
    back to `weights_only=False` (which would also re-enable arbitrary pickle
    execution), allowlist just the three PyG container classes that appear in
    the file. Harmless on older torch, which has nothing to allowlist.
    """
    names = []
    try:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        names += [DataEdgeAttr, DataTensorAttr]
    except ImportError:
        pass
    try:
        from torch_geometric.data.storage import GlobalStorage
        names += [GlobalStorage]
    except ImportError:
        pass
    if names and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(names)


def load_dataset(root="dataset"):
    """Returns (dataset, split_idx). OGB ships the scaffold split with the data."""
    _allow_pyg_globals()
    dataset = PygGraphPropPredDataset(name=DATASET, root=root)
    return dataset, dataset.get_idx_split()


def make_loaders(dataset, split_idx, batch_size=128, num_workers=0):
    out = {}
    for name in ("train", "valid", "test"):
        out[name] = DataLoader(
            dataset[split_idx[name]],
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
        )
    return out


def degree_histogram(dataset, train_idx):
    """PNA needs the in-degree histogram of the TRAINING graphs only.

    Computing it on the full dataset would leak test-set structure into the
    model's normalisation, which is the kind of leak this project avoids.
    """
    max_degree = 0
    for data in dataset[train_idx]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()) if d.numel() else 0)

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset[train_idx]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg
