"""WEGL: Wasserstein Embedding for Graph Learning, reimplemented.

Idea (Kolouri et al., ICLR 2021): skip GNN training entirely. Describe each
node by diffusing raw atom features over the bonds a few times, then treat a
molecule as a *point cloud* of node embeddings. Embed that cloud into a fixed
vector by linear optimal transport against one shared reference cloud, so two
molecules become comparable vectors and any classical classifier can take over.

This is written from the paper's description, not from the authors' code.
Two deliberate deviations, both documented in the README:
  1. PCA on node embeddings *before* the transport step. The raw one-hot
     diffusion features are ~700-dim; a reference of size R would give a
     700*R graph embedding, which does not fit in memory for 41k molecules.
  2. Exact EMD (ot.emd) with uniform weights, rather than the paper's
     sliced approximation. Molecules are small (~26 nodes), so exact is cheap.
"""
import numpy as np
import torch
from ogb.utils.features import get_atom_feature_dims
from sklearn.decomposition import PCA
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm

try:
    import ot  # POT
except ImportError:  # pragma: no cover
    ot = None


def onehot_atoms(x):
    """OGB stores 9 categorical atom features as integers. One-hot them."""
    dims = get_atom_feature_dims()
    out = np.zeros((x.shape[0], sum(dims)), dtype=np.float32)
    offset = 0
    for j, d in enumerate(dims):
        idx = np.clip(x[:, j].numpy(), 0, d - 1)
        out[np.arange(x.shape[0]), offset + idx] = 1.0
        offset += d
    return out


def diffuse(data, num_steps=3):
    """Concatenate [X, AX, A^2X, ...] with a symmetrically normalised A.

    This is the "untrained GNN" part: structure enters through how features
    mix across bonds, with no weights to learn.
    """
    x = onehot_atoms(data.x)
    n = data.num_nodes
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    A = A.multiply(dinv[:, None]).multiply(dinv[None, :]).tocsr()

    feats, cur = [x], x
    for _ in range(num_steps):
        cur = A @ cur
        feats.append(cur)
    return np.concatenate(feats, axis=1).astype(np.float32)


def node_embeddings(dataset, indices, num_steps=3, pca_dim=32, pca=None,
                    desc="diffuse", seed=0):
    """Diffusion features for every molecule, then a shared PCA to pca_dim.

    `seed` controls only the random subsample of nodes the PCA is fitted on.
    Pass a fitted `pca` to apply an existing one unchanged (valid/test).
    """
    clouds = [diffuse(dataset[int(i)], num_steps) for i in tqdm(indices, desc=desc)]
    if pca is None:
        stacked = np.concatenate(clouds, axis=0)
        sample = stacked[np.random.default_rng(seed).choice(len(stacked),
                                                            size=min(50000, len(stacked)),
                                                            replace=False)]
        pca = PCA(n_components=pca_dim, random_state=seed).fit(sample)
    return [pca.transform(c).astype(np.float32) for c in clouds], pca


def build_reference(clouds, size=30, seed=0):
    """One shared reference point cloud: a random sample of training nodes."""
    rng = np.random.default_rng(seed)
    stacked = np.concatenate(clouds, axis=0)
    idx = rng.choice(len(stacked), size=size, replace=False)
    return stacked[idx]


def lot_embed(cloud, reference):
    """Linear optimal transport embedding of one molecule.

    Transport the reference onto the molecule's nodes, take the barycentric
    projection, and record how far each reference point had to move. Two
    molecules with similar structure move the reference in similar ways.
    """
    if ot is None:
        raise ImportError("pip install POT")
    R = len(reference)
    N = len(cloud)
    a = np.full(R, 1.0 / R)
    b = np.full(N, 1.0 / N)
    M = ot.dist(reference, cloud)  # squared euclidean
    plan = ot.emd(a, b, M, numItermax=200000)
    projected = R * (plan @ cloud)  # barycentric projection
    return (projected - reference).ravel().astype(np.float32)


def embed_dataset(clouds, reference, desc="transport"):
    return np.stack([lot_embed(c, reference) for c in tqdm(clouds, desc=desc)])
