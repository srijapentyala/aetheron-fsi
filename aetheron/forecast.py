"""Forecast the NEXT flow beyond the dataset: autoregressive rollout from the last snapshot
(t = 1.50 s) through one more cardiac cycle (t = 1.52 ... 1.80 s, 15 steps). There is no ground truth here.

The surrogate needs, at each lead time, the prescribed wall state and the fluid point cloud
(AeTHERON Eq. 4 uses the structure state at t+tau). Neither exists beyond t = 1.50 s, so they come
from one period earlier (the inlet waveform is exactly 0.30 s periodic):
  fluid points (t)  := fluid points (t - 0.30 s)
  wall disp    (t)  := wall disp (t - 0.30 s) + [disp(1.50 s) - disp(1.20 s)]   (keeps cycle-to-cycle drift)

usage: python -m aetheron.forecast [--model outputs/all/model.pt]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CASES, DT, K_LAST, FSIData
from .evaluate import OUT_ROOT, PER, load_model, physical


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(OUT_ROOT, "all", "model.pt"))
    ap.add_argument("--steps", type=int, default=PER)
    a = ap.parse_args()
    model, ck = load_model(a.model)
    data = FSIData()
    data.set_stats(ck["stats"])
    out = os.path.join(OUT_ROOT, "forecast")
    os.makedirs(out, exist_ok=True)
    summary = {}
    for c in CASES:
        S = data.snaps[c]
        drift = S[K_LAST]["disp"] - S[K_LAST - PER]["disp"]
        src, fields, res = S[K_LAST], None, {}
        for j in range(1, a.steps + 1):
            k = K_LAST + j
            ref = S[k - PER]
            tgt = data.make_snapshot(ref["pos"], ref["disp"] + drift, k * DT)
            f, pb = model(data.sample(c, None, None, src=src, tgt=tgt, src_fields=fields))
            fields, src = (f, pb), tgt
            u, p = physical(data, f, pb)
            res[k] = (tgt["pos"], u, p)
            np.savez_compressed(os.path.join(out, f"case_{c}_t_{k:03d}.npz"), fluid_pos=tgt["pos"].astype(np.float32),
                                fluid_vel=u.astype(np.float32), fluid_p=p.astype(np.float32),
                                time_s=np.array(k * DT), wall_disp=(ref["disp"] + drift).astype(np.float32))
        ks = sorted(res)
        summary[c] = dict(
            t=[k * DT for k in ks],
            speed_forecast=[float(np.linalg.norm(res[k][1], axis=1).mean()) for k in ks],
            speed_prev_cycle=[float(np.linalg.norm(S[k - PER]["u"], axis=1).mean()) for k in ks],
            pbar_forecast=[float(res[k][2].mean()) for k in ks],
            pbar_prev_cycle=[float(S[k - PER]["pbar"]) for k in ks])
        # periodicity sanity check: forecast vs the same phase one cycle earlier
        rel = [np.linalg.norm(res[k][1] - S[k - PER]["u"]) / np.linalg.norm(S[k - PER]["u"]) for k in ks]
        print(f"case {c}: next-step (t={ks[0]*DT:.2f}s) mean |u| = {summary[c]['speed_forecast'][0]:.4f} m/s, "
              f"mean p = {summary[c]['pbar_forecast'][0]:.1f} Pa | cycle forecast vs previous cycle: "
              f"mean rel-L2 diff {np.mean(rel):.3f}, peak |u| {max(summary[c]['speed_forecast']):.3f} "
              f"vs {max(summary[c]['speed_prev_cycle']):.3f} m/s")
    import json
    json.dump(summary, open(os.path.join(out, "forecast_summary.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for c in CASES:
        S = data.snaps[c]
        th = [k * DT for k in sorted(S)]
        l, = axes[0].plot(th, [np.linalg.norm(S[k]["u"], axis=1).mean() for k in sorted(S)], "-", lw=1.2,
                          label=f"case {c} data")
        axes[0].plot(summary[c]["t"], summary[c]["speed_forecast"], "o--", ms=3, color=l.get_color(),
                     label=f"case {c} forecast")
        axes[1].plot(th, [S[k]["pbar"] for k in sorted(S)], "-", lw=1.2, color=l.get_color())
        axes[1].plot(summary[c]["t"], summary[c]["pbar_forecast"], "o--", ms=3, color=l.get_color())
    for ax in axes:
        ax.axvspan(K_LAST * DT, (K_LAST + a.steps) * DT, color="0.93", zorder=0)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("mean |u| (m/s)"); axes[1].set_ylabel("mean pressure (Pa)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("Forecast beyond data (shaded): mean speed")
    axes[1].set_title("Forecast beyond data (shaded): mean pressure")
    fig.tight_layout(); fig.savefig(os.path.join(out, "forecast_next_cycle.png"), dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
