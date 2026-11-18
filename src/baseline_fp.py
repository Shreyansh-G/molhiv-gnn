"""Classical baseline: Morgan fingerprints + Random Forest.

Same molecules, same scaffold split as the GNN, so the comparison is fair.
Runs on CPU in a couple of minutes and answers the question an interviewer
always asks: does the graph model actually beat counting substructures?
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

from src.data import load_dataset
from src.metrics import bootstrap_ci, rocauc

RDLogger.DisableLog("rdApp.*")


def featurize(smiles_list, n_bits=2048, radius=2):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    ok = np.ones(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            ok[i] = False
            continue
        X[i] = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.uint8)
    return X, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="cores for the forest. Defaults to half the machine so the "
                         "run leaves headroom; -1 uses every core.")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    print(f"using {args.n_jobs} of {os.cpu_count()} cores")
    os.makedirs(args.out, exist_ok=True)

    dataset, split_idx = load_dataset()
    # OGB keeps the original SMILES next to the processed graphs.
    mapping = os.path.join(dataset.root, "mapping", "mol.csv.gz")
    df = pd.read_csv(mapping)
    smiles = df["smiles"].tolist()
    y_all = df["HIV_active"].values

    X, ok = featurize(smiles)
    idx = {k: np.asarray(v) for k, v in split_idx.items()}

    # Molecules RDKit cannot parse are dropped from TRAINING only - they would
    # contribute 2048 zero bits against a real label, i.e. pure noise. Every
    # TEST molecule is kept, all-zero fingerprint included, so this arm is
    # scored on exactly the same test molecules as the WEGL and PNA arms. On
    # molhiv that is 7 unparseable SMILES: 3 in train, 2 in valid, 2 in test.
    tr = idx["train"][ok[idx["train"]]]
    te = idx["test"]
    print(f"{int((~ok).sum())} of {len(smiles)} SMILES unparseable -> "
          f"dropped {int((~ok[idx['train']]).sum())} from train "
          f"({len(tr)} left), kept all {len(te)} test molecules "
          f"({int((~ok[te]).sum())} with an all-zero fingerprint)")

    scores = []
    for seed in args.seeds:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=seed,
        )
        clf.fit(X[tr], y_all[tr])
        p = clf.predict_proba(X[te])[:, 1]
        auc = rocauc(y_all[te], p)
        lo, hi = bootstrap_ci(y_all[te], p)
        print(f"seed {seed}: Morgan+RF test ROC-AUC {auc:.4f} (95% CI {lo:.4f}-{hi:.4f})")
        scores.append(auc)
        np.savez(os.path.join(args.out, f"fp_rf_preds_seed{seed}.npz"), y=y_all[te], prob=p)

    print(f"\nMorgan+RF over {len(scores)} seeds: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
