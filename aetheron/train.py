"""Train the AeTHERON-style surrogate.

Splits
  temporal    : all 4 cases, transitions whose target is t <= 1.32 s (snapshot 66), 70/30 random
                train/val split (paper protocol); snapshots 67-75 (t = 1.34-1.50 s) are held out.
  unseen_case : train on cases 00, 06, 13 (all transitions, 85/15 train/val); case 20 is held out.
  all         : every transition of every case (90/10), used for the beyond-data forecast.

usage: python -m aetheron.train --split temporal
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import CASES, K_FIRST, K_LAST, FSIData, make_pairs
from .model import AeTHERON

SPLITS = {
    "temporal": dict(cases=CASES, k_hi=66, val_frac=0.30),
    "unseen_case": dict(cases=["00", "06", "13"], k_hi=K_LAST, val_frac=0.15),
    "all": dict(cases=CASES, k_hi=K_LAST, val_frac=0.10),
}


def loss_fn(f, pb, b):
    lu = F.l1_loss(f[:, :3], b["y"][:, :3])
    lp = F.l1_loss(f[:, 3], b["y"][:, 3])
    lpb = (pb - b["pbar_y"]).abs()
    return lu + 0.5 * lp + 0.5 * lpb, (lu.item(), lp.item(), lpb.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="temporal", choices=list(SPLITS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--dh", type=int, default=64)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.02, help="input-noise std (normalised units)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = SPLITS[args.split]
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "outputs", args.split)
    os.makedirs(out, exist_ok=True)

    data = FSIData(cases=CASES, device=args.device)
    pool = make_pairs(cfg["cases"], K_FIRST, cfg["k_hi"])
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    n_val = int(round(cfg["val_frac"] * len(pool)))
    val_pairs, train_pairs = pool[:n_val], pool[n_val:]
    stats = data.fit_stats(train_pairs)
    print(f"split={args.split}  train={len(train_pairs)}  val={len(val_pairs)}  stats={stats}")

    t0 = time.time()
    train_s = [data.sample(*p) for p in train_pairs]
    val_s = [data.sample(*p) for p in val_pairs]
    print(f"built samples in {time.time() - t0:.1f}s")

    model = AeTHERON(n_glob=FSIData.N_GLOBAL, dh=args.dh, L=args.layers).to(args.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_par/1e6:.3f}M  device={args.device}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)

    best, hist = 1e9, []
    for ep in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_s)
        tl, t0 = [], time.time()
        opt.zero_grad()
        for i, b in enumerate(train_s):
            x_in = b["x_in"] + args.noise * torch.randn_like(b["x_in"])
            pb_in = b["pbar_in"] + args.noise * torch.randn_like(b["pbar_in"])
            f, pb = model(b, x_in, pb_in)
            loss, _ = loss_fn(f, pb, b)
            (loss / args.batch).backward()
            tl.append(loss.item())
            if (i + 1) % args.batch == 0 or i + 1 == len(train_s):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = [loss_fn(*model(b), b)[1] for b in val_s]
        vu, vp, vpb = np.mean(vl, 0)
        vtot = vu + 0.5 * vp + 0.5 * vpb
        hist.append(dict(epoch=ep, train=float(np.mean(tl)), val=float(vtot), val_u=float(vu),
                         val_pp=float(vp), val_pbar=float(vpb)))
        flag = ""
        if vtot < best:
            best = vtot
            torch.save(dict(model=model.state_dict(), stats=stats, args=vars(args),
                            train_pairs=train_pairs, val_pairs=val_pairs, epoch=ep),
                       os.path.join(out, "model.pt"))
            flag = " *"
        print(f"ep {ep:3d}  train {np.mean(tl):.4f}  val {vtot:.4f} (u {vu:.4f} p' {vp:.4f} "
              f"pbar {vpb:.4f})  {time.time() - t0:.1f}s{flag}", flush=True)
    json.dump(hist, open(os.path.join(out, "train_history.json"), "w"), indent=1)
    print("best val", best)


if __name__ == "__main__":
    main()
