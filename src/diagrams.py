"""Explainer diagrams for the README, drawn from the real dataset.

    python -m src.diagrams          # writes results/figures/explain_*.png

Two ideas a newcomer needs before the results mean anything:

  explain_representations.png  one molecule, three ways of describing it
  explain_split.png            why these molecules are split by scaffold

Everything shown is computed from actual molhiv molecules - the fingerprint
bits are that molecule's real bits, the point cloud is its real diffusion
features. Nothing here is a mock-up.
"""
import argparse
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.figures import ARM_COLOR, AXIS, GRID, INK, INK_2, MUTED, SURFACE, style

RDLogger.DisableLog("rdApp.*")


def mol_image(mol, size=(420, 330), highlight=None):
    """RDKit drawing as an RGB array matplotlib can place."""
    from rdkit.Chem.Draw import rdMolDraw2D
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = d.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    if highlight:
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=highlight)
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    import PIL.Image
    return np.array(PIL.Image.open(io.BytesIO(d.GetDrawingText())).convert("RGBA"))


def panel_box(ax, title, colour, caption=None):
    """Every panel gets the identical 0..1 box, so the four line up exactly."""
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=colour, pad=10,
                 fontweight="bold")
    if caption:
        ax.text(0.5, -0.11, caption, transform=ax.transAxes, ha="center",
                va="top", fontsize=9, color=INK_2, linespacing=1.5)


def _unit(pts, pad=0.12):
    """Scale coordinates into the shared 0..1 panel box."""
    pts = np.asarray(pts, dtype=float)
    lo, hi = pts.min(0), pts.max(0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    return pad + (pts - lo) / span * (1 - 2 * pad)


def fig_representations(smiles, out):
    """One molecule -> a checklist, a cloud of atoms, a graph."""
    mol = Chem.MolFromSmiles(smiles)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.3))
    ax0, ax1, ax2, ax3 = axes

    # --- the molecule itself -------------------------------------------
    ax0.imshow(mol_image(mol, size=(460, 400)), extent=(0, 1, 0, 1),
               aspect="auto", interpolation="antialiased")
    panel_box(ax0, "A molecule", INK,
              f"{mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")

    # --- 1. fingerprint: which fragments are present --------------------
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bits = np.asarray(gen.GetFingerprintAsNumPy(mol))
    grid = bits.reshape(32, 64)
    ax1.imshow(grid, extent=(0, 1, 0, 1), aspect="auto",
               interpolation="nearest",
               cmap=matplotlib.colors.ListedColormap(["#ededeb", ARM_COLOR["fp"]]))
    panel_box(ax1, "1 · A checklist", ARM_COLOR["fp"],
              f"{int(bits.sum())} of 2048 boxes ticked\nnothing is learned")

    # --- 2. WEGL: atoms as a cloud of descriptions ----------------------
    from src.wegl import diffuse
    from torch_geometric.utils import from_smiles
    from sklearn.decomposition import PCA
    data = from_smiles(smiles)
    feats = diffuse(data, num_steps=3)
    pts = _unit(PCA(n_components=2, random_state=0).fit_transform(feats))
    ax2.scatter(pts[:, 0], pts[:, 1], s=120, color=ARM_COLOR["wegl"],
                edgecolor=SURFACE, linewidth=1.8, zorder=3)
    panel_box(ax2, "2 · A cloud of atoms", ARM_COLOR["wegl"],
              "each atom described by what\nsurrounds it; nothing is learned")

    # --- 3. message passing ---------------------------------------------
    pos = _unit(_ring_layout(mol))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ax3.plot(*zip(pos[i], pos[j]), color=AXIS, lw=1.8, zorder=1)
    for b in list(mol.GetBonds())[:8]:
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ax3.add_patch(FancyArrowPatch(pos[i], pos[j], arrowstyle="-|>",
                                      mutation_scale=13, color=ARM_COLOR["pna"],
                                      lw=2, zorder=2, shrinkA=8, shrinkB=8))
    ax3.scatter(pos[:, 0], pos[:, 1], s=110, color=SURFACE,
                edgecolor=ARM_COLOR["pna"], linewidth=2.2, zorder=3)
    panel_box(ax3, "3 · Messages along bonds", ARM_COLOR["pna"],
              "atoms share information with\ntheir neighbours, and it is learned")

    fig.suptitle("The same molecule, described three ways",
                 x=0.012, ha="left", fontsize=13.5)
    fig.text(0.012, 0.025,
             "Only the third one learns anything from the data. That contrast is "
             "what this project is built around.", color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(os.path.join(out, "explain_representations.png"))
    plt.close(fig)


def _ring_layout(mol):
    """A readable 2D layout for the graph panel."""
    from rdkit.Chem import AllChem
    m = Chem.Mol(mol)
    AllChem.Compute2DCoords(m)
    c = m.GetConformer()
    return np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                     for i in range(m.GetNumAtoms())])


def fig_split(smiles_list, out):
    """Why the molecules are grouped by skeleton before splitting."""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles_list[0])
    mols = [Chem.MolFromSmiles(s) for s in smiles_list[:4]]
    core = Chem.MolFromSmiles(scaffold)

    fig = plt.figure(figsize=(12.4, 5.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.78], hspace=0.08, wspace=0.10,
                          top=0.84, bottom=0.04, left=0.012, right=0.988)

    for k, m in enumerate(mols):
        ax = fig.add_subplot(gs[0, k])
        match = m.GetSubstructMatch(core) if core else ()
        ax.imshow(mol_image(m, size=(400, 290), highlight=list(match)),
                  extent=(0, 1, 0, 1), aspect="auto",
                  interpolation="antialiased")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.012, 0.945, "Four different molecules that share one skeleton",
             fontsize=13.5, color=INK)
    fig.text(0.012, 0.895,
             "The highlighted core is their Bemis-Murcko scaffold. Chemically "
             "they are close relatives.", fontsize=9.5, color=INK_2)

    # --- the two ways of splitting them --------------------------------
    axl = fig.add_subplot(gs[1, :2])
    axr = fig.add_subplot(gs[1, 2:])
    for ax, title, assign, note, tone in [
        (axl, "Split at random", ["train", "test", "train", "test"],
         "Close relatives land on both sides. The model half-recognises the\n"
         "test set, and every score comes out looking better than it is.",
         "#d03b3b"),
        (axr, "Split by scaffold  ← used here",
         ["train", "train", "train", "train"],
         "The whole family goes to one side. Test molecules are built on\n"
         "skeletons the model has genuinely never seen.", "#0ca30c"),
    ]:
        ax.set_xlim(0, 4); ax.set_ylim(0, 1.5)
        ax.axis("off")
        ax.text(0, 1.38, title, fontsize=11.5, color=tone, fontweight="bold")
        for i, a in enumerate(assign):
            col = ARM_COLOR["fp"] if a == "train" else ARM_COLOR["wegl"]
            ax.add_patch(FancyBboxPatch((i + 0.12, 0.70), 0.76, 0.42,
                                        boxstyle="round,pad=0.02,rounding_size=0.06",
                                        facecolor=col, edgecolor="none"))
            ax.text(i + 0.5, 0.91, a, ha="center", va="center", color="#ffffff",
                    fontsize=10, fontweight="bold")
            ax.text(i + 0.5, 1.20, f"molecule {i+1}", ha="center", fontsize=8.5,
                    color=MUTED)
        ax.text(0, 0.30, note, fontsize=9.5, color=INK_2, va="top")

    fig.savefig(os.path.join(out, "explain_split.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    style()

    import pandas as pd
    from src.data import load_dataset
    dataset, _ = load_dataset()
    df = pd.read_csv(os.path.join(dataset.root, "mapping", "mol.csv.gz"))
    smiles = df["smiles"].tolist()

    # a mid-sized active molecule for the representation diagram
    actives = df[df["HIV_active"] == 1]["smiles"].tolist()
    pick = min((s for s in actives if 14 <= Chem.MolFromSmiles(s).GetNumAtoms() <= 22
                if Chem.MolFromSmiles(s)), key=lambda s: abs(
                    Chem.MolFromSmiles(s).GetNumAtoms() - 18))
    print("representation diagram uses:", pick)
    fig_representations(pick, args.out)
    print("  explain_representations.png")

    # a scaffold shared by several molecules, for the split diagram
    groups = {}
    for s in smiles[:6000]:
        m = Chem.MolFromSmiles(s)
        if m is None or not (16 <= m.GetNumAtoms() <= 26):
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(s)
        except Exception:
            continue
        if sc and Chem.MolFromSmiles(sc) and \
                Chem.MolFromSmiles(sc).GetNumAtoms() >= 10:
            groups.setdefault(sc, []).append(s)
    family = max(groups.values(), key=len)
    print(f"split diagram uses a scaffold shared by {len(family)} molecules")
    fig_split(family, args.out)
    print("  explain_split.png")
    print(f"\nwrote diagrams to {args.out}")


if __name__ == "__main__":
    main()
