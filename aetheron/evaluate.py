"""Evaluate a trained model: one-step ("next flow") prediction and autoregressive rollout,
against honest baselines.

Baselines
  persistence   : next flow = current flow (interpolated onto the next point cloud)
  prev_cycle    : next flow = flow one cardiac period earlier (t - 0.30 s), interpolated
  other_cases   : (unseen-case split only) mean of the training cases' flow at the same time

usage: python -m aetheron.evaluate --split temporal
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CASES, DT, K_FIRST, K_LAST, FSIData, idw_apply, idw_weights
from .model import AeTHERON

PER = 15   # snapshots per cardiac cycle
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

TESTS = {
    "temporal": dict(cases=CASES, onestep=list(range(67, K_LAST + 1)), roll_start=66),
    "unseen_case": dict(cases=["20"], onestep=list(range(K_FIRST + 1, K_LAST + 1)), roll_start=K_FIRST,
                        ref_cases=["00", "06", "13"]),
}


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    m = AeTHERON(n_glob=FSIData.N_GLOBAL, dh=a["dh"], L=a["layers"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


@torch.no_grad()
def predict(model, data, c, ks, kt, src_fields=None, src=None, tgt=None):
    b = data.sample(c, ks, kt, src=src, tgt=tgt, src_fields=src_fields)
    f, pb = model(b)
    return f, pb


def physical(data, f, pb):
    u, pp, pbar = data.decode_fields(f, pb)
    return u, pp + pbar


def metrics(u_pred, p_pred, T):
    du = u_pred - T["u"]
    p_true = T["pprime"] + T["pbar"]
    pp_pred = p_pred - p_pred.mean()
    return dict(
        vel_mae=float(np.abs(du).mean()),
        vel_rel_l2=float(np.linalg.norm(du) / np.linalg.norm(T["u"])),
        speed_mae=float(np.abs(np.linalg.norm(u_pred, axis=1) - np.linalg.norm(T["u"], axis=1)).mean()),
        p_mae=float(np.abs(p_pred - p_true).mean()),
        pprime_mae=float(np.abs(pp_pred - T["pprime"]).mean()),
        pbar_err=float(p_pred.mean() - T["pbar"]))


def interp_truth(data, c_src, k_src, T):
    S = data.snaps[c_src][k_src]
    idx, w = idw_weights(S["pos"], T["pos"])
    return idw_apply(idx, w, S["u"]), idw_apply(idx, w, S["pprime"][:, None])[:, 0] + S["pbar"]


def evaluate_split(split, device="cpu"):
    cfg = TESTS[split]
    out = os.path.join(OUT_ROOT, split)
    model, ck = load_model(os.path.join(out, "model.pt"), device)
    data = FSIData(device=device)
    data.set_stats(ck["stats"])
    rows, preds = [], {}
    for c in cfg["cases"]:
        # ---- one-step: true state at k-1 -> predict k
        for kt in cfg["onestep"]:
            T = data.snaps[c][kt]
            f, pb = predict(model, data, c, kt - 1, kt)
            u, p = physical(data, f, pb)
            rows.append(dict(case=c, k=kt, t=T["time"], mode="one_step", method="AeTHERON", **metrics(u, p, T)))
            preds[f"onestep_{c}_{kt}"] = np.concatenate([u, p[:, None]], 1).astype(np.float32)
            u, p = interp_truth(data, c, kt - 1, T)
            rows.append(dict(case=c, k=kt, t=T["time"], mode="one_step", method="persistence", **metrics(u, p, T)))
            if kt - PER >= K_FIRST:
                u, p = interp_truth(data, c, kt - PER, T)
                rows.append(dict(case=c, k=kt, t=T["time"], mode="one_step", method="prev_cycle", **metrics(u, p, T)))
            if "ref_cases" in cfg:
                us, ps = zip(*[interp_truth(data, r, kt, T) for r in cfg["ref_cases"]])
                rows.append(dict(case=c, k=kt, t=T["time"], mode="one_step", method="other_cases_same_t",
                                 **metrics(np.mean(us, 0), np.mean(ps, 0), T)))
        # ---- autoregressive rollout from the true state at roll_start
        k0 = cfg["roll_start"]
        fields = None
        for kt in range(k0 + 1, K_LAST + 1):
            T = data.snaps[c][kt]
            f, pb = predict(model, data, c, kt - 1, kt, src_fields=fields)
            fields = (f, pb)
            u, p = physical(data, f, pb)
            rows.append(dict(case=c, k=kt, t=T["time"], mode="rollout", method="AeTHERON", **metrics(u, p, T)))
            preds[f"rollout_{c}_{kt}"] = np.concatenate([u, p[:, None]], 1).astype(np.float32)
            u, p = interp_truth(data, c, k0, T)
            rows.append(dict(case=c, k=kt, t=T["time"], mode="rollout", method="persistence", **metrics(u, p, T)))
    json.dump(rows, open(os.path.join(out, "metrics_per_snapshot.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out, "predictions.npz"), **preds)

    summary = {}
    for mode in ["one_step", "rollout"]:
        for meth in sorted({r["method"] for r in rows if r["mode"] == mode}):
            R = [r for r in rows if r["mode"] == mode and r["method"] == meth]
            summary[f"{mode}/{meth}"] = {k: float(np.mean([r[k] for r in R]))
                                         for k in ["vel_mae", "vel_rel_l2", "speed_mae", "p_mae", "pprime_mae"]}
            summary[f"{mode}/{meth}"]["n"] = len(R)
    # like-for-like comparison against prev_cycle on the snapshots where it exists
    ks_pc = {(r["case"], r["k"]) for r in rows if r["method"] == "prev_cycle"}
    if ks_pc:
        for meth in ["AeTHERON", "persistence", "prev_cycle"]:
            R = [r for r in rows if r["mode"] == "one_step" and r["method"] == meth and (r["case"], r["k"]) in ks_pc]
            summary[f"one_step_on_prev_cycle_subset/{meth}"] = {
                k: float(np.mean([r[k] for r in R])) for k in ["vel_mae", "vel_rel_l2", "p_mae"]}
    json.dump(summary, open(os.path.join(out, "metrics_summary.json"), "w"), indent=1)
    print(f"\n=== {split} ===")
    for k, v in summary.items():
        print(f"{k:45s} " + "  ".join(f"{m}={x:.4g}" for m, x in v.items()))
    make_figures(split, data, rows, preds, cfg)
    return summary


# ---------------------------------------------------------------------- figures
def _slab(data, pos, width=0.6e-3):
    c = data.wall_pos0 - data.center
    w, v = np.linalg.eigh(np.cov(c.T))
    n, a1, a2 = v[:, 0], v[:, 2], v[:, 1]
    r = pos - data.center
    m = np.abs(r @ n) < width
    return m, r[m] @ a1 * 1e3, r[m] @ a2 * 1e3


def make_figures(split, data, rows, preds, cfg):
    out = os.path.join(OUT_ROOT, split)
    colors = {"AeTHERON": "#c2410c", "persistence": "#64748b", "prev_cycle": "#2563eb",
              "other_cases_same_t": "#16a34a"}
    # error vs time
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, mode in zip(axes, ["one_step", "rollout"]):
        for meth in colors:
            R = [r for r in rows if r["mode"] == mode and r["method"] == meth]
            if not R:
                continue
            ks = sorted({r["k"] for r in R})
            y = [np.mean([r["vel_mae"] for r in R if r["k"] == k]) for k in ks]
            ax.plot(np.array(ks) * DT, y, "o-", ms=3, color=colors[meth], label=meth)
        ax2 = ax.twinx()
        tt = np.linspace(min(r["t"] for r in rows) - DT, K_LAST * DT, 300)
        ax2.fill_between(tt, data.inlet(tt), color="0.85", zorder=0)
        ax2.set_ylabel("inlet velocity (m/s)", color="0.5")
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_xlabel("time (s)"); ax.set_title(f"{mode.replace('_', '-')} velocity MAE")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("velocity MAE (m/s)")
    fig.suptitle(f"{split}: cases {', '.join(cfg['cases'])}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "error_vs_time.png"), dpi=140); plt.close(fig)

    # slices at peak systole within the test window
    c = cfg["cases"][-1]
    cand = [k for k in cfg["onestep"]]
    kpk = max(cand, key=lambda k: np.abs(data.snaps[c][k]["u"]).mean())
    T = data.snaps[c][kpk]
    m, X, Y = _slab(data, T["pos"])
    sp_t = np.linalg.norm(T["u"], axis=1)[m]
    panels = [("ground truth |u|", sp_t)]
    for key, lab in [(f"onestep_{c}_{kpk}", "AeTHERON one-step"), (f"rollout_{c}_{kpk}", "AeTHERON rollout")]:
        sp = np.linalg.norm(preds[key][:, :3], axis=1)[m]
        panels += [(lab + " |u|", sp), (lab + " |error|", np.abs(sp - sp_t))]
    S = data.snaps[c][kpk - 1]
    idx, w = idw_weights(S["pos"], T["pos"])
    sp_pers = np.linalg.norm(idw_apply(idx, w, S["u"]), axis=1)[m]
    panels += [("persistence |u|", sp_pers), ("persistence |error|", np.abs(sp_pers - sp_t))]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    vmax = sp_t.max()
    order = [0, 1, 3, 5, None, 2, 4, 6]
    for ax, i in zip(axes.flat, order):
        if i is None:
            ax.axis("off"); continue
        lab, val = panels[i]
        err = "error" in lab
        sc = ax.scatter(X, Y, c=val, s=9, cmap="magma" if err else "viridis", vmin=0,
                        vmax=vmax * (0.5 if err else 1))
        ax.set_title(lab, fontsize=10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, label="m/s")
    fig.suptitle(f"{split}: case {c}, t = {T['time']:.2f} s (peak flow in test window), mid-plane slab")
    fig.tight_layout(); fig.savefig(os.path.join(out, "slice_peak.png"), dpi=130); plt.close(fig)

    # pressure level
    fig, ax = plt.subplots(figsize=(7, 4))
    for cc in cfg["cases"]:
        R = [r for r in rows if r["mode"] == "rollout" and r["method"] == "AeTHERON" and r["case"] == cc]
        ks = [r["k"] for r in R]
        ax.plot(np.array(ks) * DT, [data.snaps[cc][k]["pbar"] for k in ks], "-", label=f"case {cc} true")
        ax.plot(np.array(ks) * DT, [data.snaps[cc][k]["pbar"] + r["pbar_err"] for k, r in zip(ks, R)],
                "--", color=ax.lines[-1].get_color(), label=f"case {cc} rollout")
    ax.set_xlabel("time (s)"); ax.set_ylabel("mean pressure (Pa)"); ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{split}: domain-mean pressure, autoregressive rollout")
    fig.tight_layout(); fig.savefig(os.path.join(out, "pressure_rollout.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="temporal", choices=list(TESTS))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    evaluate_split(a.split, a.device)
