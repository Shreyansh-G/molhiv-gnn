"""Which molecules does the model get wrong, and do they have anything in common?

Draws the most confident mistakes as a PNG grid. This is the part a chemist
will actually look at, and the question interviewers ask after the metrics.
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, Descriptors

from src.data import load_dataset

RDLogger.DisableLog("rdApp.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/test_preds_seed0.npz")
    ap.add_argument("--k", type=int, default=6, help="molecules per panel")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    d = np.load(args.preds)
    y = d["y"].ravel()
    prob = d["prob"].ravel() if "prob" in d else 1 / (1 + np.exp(-d["logit"].ravel()))

    dataset, split_idx = load_dataset()
    df = pd.read_csv(os.path.join(dataset.root, "mapping", "mol.csv.gz"))
    smiles = np.asarray(df["smiles"].tolist())[np.asarray(split_idx["test"])]

    fp = np.where(y == 0)[0][np.argsort(-prob[y == 0])][: args.k]   # confident, wrong
    fn = np.where(y == 1)[0][np.argsort(prob[y == 1])][: args.k]    # missed actives

    rows = []
    for name, idx in (("false_positive", fp), ("false_negative", fn)):
        mols, legends = [], []
        for i in idx:
            mol = Chem.MolFromSmiles(smiles[i])
            if mol is None:
                continue
            mols.append(mol)
            legends.append(f"p={prob[i]:.2f}")
            rows.append({
                "type": name,
                "smiles": smiles[i],
                "label": int(y[i]),
                "prob": float(prob[i]),
                "mol_weight": Descriptors.MolWt(mol),
                "num_rings": Descriptors.RingCount(mol),
                "logp": Descriptors.MolLogP(mol),
            })
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 250), legends=legends)
            path = os.path.join(args.out, f"{name}s.png")
            # What MolsToGridImage hands back depends on the RDKit build and on
            # whether IPython is importable: a PIL Image (RDKit 2026.09), an
            # IPython Image wrapping PNG bytes, or raw bytes/str. Assuming one
            # of them wrote a 0-byte PNG and then killed the script before the
            # CSV, so handle all three.
            if hasattr(img, "save"):                       # PIL Image
                img.save(path)
            else:
                data = getattr(img, "data", img)
                if isinstance(data, str):                  # SVG text
                    data = data.encode()
                with open(path, "wb") as f:
                    f.write(data)
            print(f"wrote {path}")

    out_csv = os.path.join(args.out, "error_analysis.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    summary = pd.DataFrame(rows).groupby("type")[["mol_weight", "num_rings", "logp"]].mean()
    print("\nmean properties of the worst mistakes:\n", summary.round(2))


if __name__ == "__main__":
    main()
