"""Train the WEGL arm: graph embeddings (no learning) + a classical classifier.

Builds the Wasserstein embedding (PCA + reference cloud fitted on TRAIN only),
then fits a classical classifier on it. The number of boosting iterations is
chosen on the official scaffold validation split, the same molecules and the
same protocol the PNA arm uses to pick its best epoch.

    python -m src.run_wegl --seeds 0 1 2 --classifier hgb
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.data import load_dataset
from src.metrics import bootstrap_ci, rocauc
from src.wegl import build_reference, embed_dataset, node_embeddings


def build_embeddings(dataset, split_idx, args, seed):
    """Diffuse -> PCA -> optimal-transport embedding, cached per hyperparameter set.

    The cache filename carries every setting that changes the embedding, so
    re-running with a different --pca-dim cannot silently reuse the old one.
    """
    tag = (f"s{args.diffusion_steps}_p{args.pca_dim}_r{args.reference_size}"
           f"_seed{seed}")
    cache = None if args.no_cache else os.path.join(
        args.out, f"wegl_cache_{tag}.pkl")
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            print(f"loaded cached embeddings from {cache}")
            return pickle.load(f)

    # PCA and the reference cloud are fitted on TRAIN ONLY, then applied
    # unchanged to valid/test. Fitting them on everything would leak.
    train_clouds, pca = node_embeddings(
        dataset, split_idx["train"], args.diffusion_steps, args.pca_dim,
        desc="diffuse train", seed=seed,
    )
    reference = build_reference(train_clouds, size=args.reference_size, seed=seed)
    emb = {"train": embed_dataset(train_clouds, reference, "transport train")}
    del train_clouds
    for part in ("valid", "test"):
        clouds, _ = node_embeddings(
            dataset, split_idx[part], args.diffusion_steps, args.pca_dim,
            pca=pca, desc=f"diffuse {part}", seed=seed,
        )
        emb[part] = embed_dataset(clouds, reference, f"transport {part}")
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(emb, f)
        print(f"cached embeddings to {cache} "
              f"({os.path.getsize(cache) / 1e6:.0f} MB)")
    return emb


def fit_rf(emb, y_tr, seed, balanced, n_jobs=-1):
    clf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced" if balanced else None,
        n_jobs=n_jobs,
        random_state=seed,
    )
    clf.fit(emb["train"], y_tr)
    return clf.predict_proba(emb["test"])[:, 1], {"n_estimators": 500}


def fit_hgb(emb, y_tr, y_va, seed, balanced, max_iter=300, step=25, patience=3):
    """Boosted trees whose iteration count is chosen on the official valid split.

    sklearn's HistGradientBoostingClassifier defaults to early_stopping="auto",
    which on >10k rows carves a *random* 10% of training molecules out as its
    own validation set. On a scaffold benchmark that is the wrong held-out set:
    a random slice shares scaffolds with the rows it was cut from, so it says
    nothing about new chemistry, and on a 3.5%-positive target it may hold only
    a handful of actives. Here it is switched off and the iteration count is
    selected on the official scaffold validation split instead - the same
    molecules, and the same "best validation checkpoint" rule, that the PNA arm
    uses. The sweep below never looks at the test set: it scores validation
    only, then refits once at the chosen iteration count and predicts test
    exactly once, so "the test set is touched once per arm per seed" is
    literally true.
    """
    common = dict(
        learning_rate=0.1,
        class_weight="balanced" if balanced else None,
        early_stopping=False,
        random_state=seed,
    )
    sweep = HistGradientBoostingClassifier(max_iter=step, warm_start=True, **common)
    best_auc, best_n, curve, stale = -1.0, step, [], 0
    for n in range(step, max_iter + 1, step):
        sweep.set_params(max_iter=n)
        sweep.fit(emb["train"], y_tr)
        va = rocauc(y_va, sweep.predict_proba(emb["valid"])[:, 1])
        curve.append((n, round(va, 4)))
        if va > best_auc:
            best_auc, best_n, stale = va, n, 0
        else:
            # Stop once validation ROC-AUC has failed to improve for `patience`
            # consecutive checkpoints, the same rule as early stopping on the
            # PNA arm. This is what keeps the sweep affordable: sklearn's own
            # (random-split) early stopping settles after a few dozen
            # iterations on this embedding, so walking all the way to max_iter
            # buys nothing and costs 10x the time.
            stale += 1
            if stale >= patience:
                break

    final = HistGradientBoostingClassifier(max_iter=best_n, **common)
    final.fit(emb["train"], y_tr)
    return (final.predict_proba(emb["test"])[:, 1],
            {"best_iter": best_n, "valid_auc": round(best_auc, 4),
             "searched_to": curve[-1][0], "valid_curve": curve})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--classifier", choices=["rf", "hgb"], default="rf",
                    help="rf (default) fits 500 trees in ~90s. hgb is the boosted "
                         "ablation and costs ~12s per boosting iteration on this "
                         "960-dim embedding, so budget hours per seed.")
    ap.add_argument("--diffusion-steps", type=int, default=3)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--reference-size", type=int, default=30)
    ap.add_argument("--max-iter", type=int, default=300,
                    help="hgb: largest iteration count considered")
    ap.add_argument("--sweep-step", type=int, default=25,
                    help="hgb: checkpoint spacing when selecting the iteration count")
    ap.add_argument("--sweep-patience", type=int, default=3,
                    help="hgb: stop the sweep after this many checkpoints with no "
                         "validation improvement")
    ap.add_argument("--no-balance", action="store_true", help="drop class weighting")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not write the embedding cache. It is ~160 MB per seed "
                         "and only ~2 min to rebuild; caching pays off when trying "
                         "several classifiers on one embedding, not otherwise.")
    ap.add_argument("--share-embedding", action="store_true",
                    help="build the embedding once (seed 0) and reuse it for every "
                         "seed. Then the reported std covers classifier randomness "
                         "only, not the random reference cloud.")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="cores for the Random Forest. Defaults to half the machine; "
                         "-1 uses every core. The boosted-tree path is OpenMP-based, "
                         "so cap it with OMP_NUM_THREADS instead.")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"using {args.n_jobs} of {os.cpu_count()} cores "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")

    dataset, split_idx = load_dataset()
    y = dataset.y.numpy().ravel()
    y_tr = y[np.asarray(split_idx["train"])]
    y_va = y[np.asarray(split_idx["valid"])]
    y_te = y[np.asarray(split_idx["test"])]

    scores, infos = [], []
    for seed in args.seeds:
        # By default the embedding is rebuilt per seed, so the seed-to-seed std
        # includes the random reference-cloud draw rather than hiding it.
        emb = build_embeddings(dataset, split_idx, args,
                               seed=0 if args.share_embedding else seed)
        if seed == args.seeds[0]:
            print(f"embedding dim: {emb['train'].shape[1]}  "
                  f"(train {emb['train'].shape[0]}, valid {emb['valid'].shape[0]}, "
                  f"test {emb['test'].shape[0]})")

        balanced = not args.no_balance
        if args.classifier == "rf":
            p, info = fit_rf(emb, y_tr, seed, balanced, n_jobs=args.n_jobs)
        else:
            p, info = fit_hgb(emb, y_tr, y_va, seed, balanced,
                              max_iter=args.max_iter, step=args.sweep_step,
                              patience=args.sweep_patience)

        auc = rocauc(y_te, p)
        lo, hi = bootstrap_ci(y_te, p)
        curve = info.pop("valid_curve", None)
        if curve:
            print(f"  seed {seed} valid AUC by iteration: "
                  + " ".join(f"{n}:{v:.4f}" for n, v in curve))
        print(f"seed {seed}: WEGL+{args.classifier} test ROC-AUC {auc:.4f} "
              f"(95% CI {lo:.4f}-{hi:.4f})  {info}")
        scores.append(auc)
        infos.append(info)
        np.savez(os.path.join(args.out, f"wegl_preds_seed{seed}.npz"), y=y_te, prob=p)

    print(f"\nWEGL+{args.classifier} over {len(scores)} seeds: "
          f"{np.mean(scores):.4f} +/- {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
