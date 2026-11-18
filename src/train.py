"""Train PNA (or the OGB GIN baseline shape) on ogbg-molhiv over several seeds.

Usage:
    python -m src.train --seeds 0 1 2 --epochs 50
Writes per-seed test predictions to results/ and prints mean +/- std ROC-AUC.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.data import degree_histogram, load_dataset, make_loaders
from src.metrics import bootstrap_ci, calibration, rocauc
from src.models import PNANet


def run_epoch(model, loader, device, optimizer=None, pos_weight=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    ys, preds = [], []
    for batch in loader:
        batch = batch.to(device)
        if train and batch.x.shape[0] == 1:
            # BatchNorm1d cannot compute a batch variance from a single node,
            # but only while training; in eval() it uses running stats and is
            # fine. Skipping in eval would silently drop a molecule from the
            # test predictions and score this arm on a different test set from
            # the other two. (On molhiv this never fires - the smallest
            # molecule has 2 atoms - but the guard belongs in train only.)
            continue
        with torch.set_grad_enabled(train):
            out = model(batch)
            target = batch.y.to(torch.float32)
            mask = target == target  # OGB marks missing labels as NaN
            loss = F.binary_cross_entropy_with_logits(
                out[mask], target[mask], pos_weight=pos_weight
            )
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss) * batch.num_graphs
        n += batch.num_graphs
        ys.append(target.detach().cpu())
        preds.append(out.detach().cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(preds).numpy()
    return total_loss / max(n, 1), y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=80)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=3e-6)
    ap.add_argument("--pos-weight", type=float, default=0.0,
                    help="weight on the positive class in the loss; 0 = unweighted. "
                         "Set e.g. 10 to counter the 3.5%% positive rate.")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset, split_idx = load_dataset()
    loaders = make_loaders(dataset, split_idx, batch_size=args.batch_size)
    deg = degree_histogram(dataset, split_idx["train"])
    print(f"degree histogram computed on {len(split_idx['train'])} training molecules")

    test_scores = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PNANet(
            deg=deg,
            emb_dim=args.emb_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # mode="max": the metric we step with is validation ROC-AUC, where
        # higher is better. ReduceLROnPlateau defaults to mode="min", which
        # treats a *rising* AUC as "no improvement" and cuts the learning rate
        # while the model is still getting better.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        pw = torch.tensor(args.pos_weight, device=device) if args.pos_weight > 0 else None
        best_valid, best_state = -1.0, None
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            loss, _, _ = run_epoch(model, loaders["train"], device, optimizer, pos_weight=pw)
            _, yv, pv = run_epoch(model, loaders["valid"], device)
            valid_auc = rocauc(yv, pv)
            scheduler.step(valid_auc)
            if valid_auc > best_valid:
                best_valid = valid_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == 1:
                print(f"  seed {seed} epoch {epoch:3d} loss {loss:.4f} valid {valid_auc:.4f}")

        model.load_state_dict(best_state)
        _, yt, pt = run_epoch(model, loaders["test"], device)
        test_auc = rocauc(yt, pt)
        test_scores.append(test_auc)
        probs = torch.sigmoid(torch.tensor(pt)).numpy()
        lo, hi = bootstrap_ci(yt, pt)
        print(
            f"seed {seed}: test ROC-AUC {test_auc:.4f} (95% CI {lo:.4f}-{hi:.4f}), "
            f"valid {best_valid:.4f}, {n_params:,} params, {time.time()-t0:.0f}s"
        )
        torch.save({"state_dict": best_state, "deg": deg, "args": vars(args)},
                   os.path.join(args.out, f"pna_seed{seed}.pt"))
        np.savez(os.path.join(args.out, f"test_preds_seed{seed}.npz"), y=yt, logit=pt, prob=probs)
        with open(os.path.join(args.out, f"calibration_seed{seed}.json"), "w") as f:
            json.dump(calibration(yt, probs), f, indent=2)

    print(f"\nPNA over {len(test_scores)} seeds: "
          f"{np.mean(test_scores):.4f} +/- {np.std(test_scores):.4f}")


if __name__ == "__main__":
    main()
