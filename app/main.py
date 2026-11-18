"""FastAPI service: SMILES in, HIV-inhibition probability out.

Loads every checkpoint in results/ as an ensemble, so each prediction also
carries a spread across models - the closest thing to an honest uncertainty
estimate without retraining.
"""
import glob
import os
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from ogb.utils.mol import smiles2graph
from pydantic import BaseModel, Field
from torch_geometric.data import Batch, Data

from src.models import PNANet

CKPT_GLOB = os.environ.get("CKPT_GLOB", "results/pna_seed*.pt")
app = FastAPI(title="molhiv-gnn", version="0.1.0")
_models: List[PNANet] = []


class Query(BaseModel):
    smiles: List[str] = Field(..., examples=[["CCO", "CC(=O)Oc1ccccc1C(=O)O"]])


def _to_data(smi: str) -> Data:
    g = smiles2graph(smi)
    return Data(
        x=torch.from_numpy(g["node_feat"]).to(torch.long),
        edge_index=torch.from_numpy(g["edge_index"]).to(torch.long),
        edge_attr=torch.from_numpy(g["edge_feat"]).to(torch.long),
    )


@app.on_event("startup")
def load_models():
    for path in sorted(glob.glob(CKPT_GLOB)):
        ckpt = torch.load(path, map_location="cpu")
        a = ckpt["args"]
        model = PNANet(
            deg=ckpt["deg"],
            emb_dim=a["emb_dim"],
            num_layers=a["num_layers"],
            dropout=a["dropout"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _models.append(model)
    print(f"loaded {len(_models)} checkpoints from {CKPT_GLOB}")


@app.get("/health")
def health():
    return {"status": "ok", "models": len(_models)}


@app.post("/predict")
def predict(q: Query):
    if not _models:
        raise HTTPException(503, "no checkpoints loaded; train a model first")
    try:
        batch = Batch.from_data_list([_to_data(s) for s in q.smiles])
    except Exception as exc:
        raise HTTPException(400, f"could not parse SMILES: {exc}")

    with torch.no_grad():
        probs = np.stack([torch.sigmoid(m(batch)).numpy().ravel() for m in _models])

    return {
        "predictions": [
            {
                "smiles": s,
                "probability": float(probs[:, i].mean()),
                "ensemble_std": float(probs[:, i].std()),
            }
            for i, s in enumerate(q.smiles)
        ]
    }
