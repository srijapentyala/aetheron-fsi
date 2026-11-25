"""Data pipeline for the FSI student subset, framed as in AeTHERON (arXiv:2604.13369).

Each sample is a heterogeneous graph pair for a transition  t_k  ->  t_{k+tau}:
  * fluid graph   : nodes = fluid point cloud of the TARGET snapshot (solver remeshes every step,
                    so fluid node indices are not persistent). Input fields at time t_k are moved
                    onto those nodes by inverse-distance interpolation from the source cloud.
  * wall graph    : the 4246 wall nodes (rows persist across snapshots), features use the wall
                    state at lead time t_{k+tau} (prescribed structure, as in the paper, Eq. 4).
  * wall->fluid   : each fluid node is linked to its K_WF nearest (deformed) wall nodes.
"""
import glob
import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = ["00", "06", "13", "20"]
K_FIRST, K_LAST = 45, 75          # snapshot indices on disk (t = k * 0.02 s)
DT = 0.02                         # s between snapshots
PERIOD = 0.30                     # s, cardiac period of the shared inlet waveform (15 snapshots)
L_SCALE = 1e-3                    # m, length scale for coordinates (mm)
H_SCALE = 2.5e-4                  # m, typical fluid point spacing, scales edge vectors
K_FF, K_WW, K_WF, K_INTERP = 8, 8, 8, 4


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def idw_weights(src_pos, tgt_pos, k=K_INTERP):
    d, i = cKDTree(src_pos).query(tgt_pos, k=k)
    w = 1.0 / (d + 1e-9) ** 2
    w /= w.sum(1, keepdims=True)
    return i.astype(np.int64), w.astype(np.float32)


def idw_apply(idx, w, f):
    """f: (N_src, C) array/tensor -> (N_tgt, C)"""
    return (f[idx] * w[..., None]).sum(1)


class FSIData:
    def __init__(self, cases=CASES, device="cpu"):
        self.cases = list(cases)
        self.device = device
        meta = json.load(open(os.path.join(ROOT, "subset_metadata.json")))
        # R1, R2, C1, C2, Z1, Z2 are all fixed multiples of s, so (mu, alpha, s) are the free parameters
        self.params = {
            c: np.array([meta["case_parameters"][c]["μ (Pa)"],
                         meta["case_parameters"][c]["α"],
                         meta["case_parameters"][c]["s"]], dtype=np.float64)
            for c in CASES}
        P = np.stack([self.params[c] for c in CASES])
        self.param_mu, self.param_sd = P.mean(0), P.std(0) + 1e-12

        inl = np.loadtxt(os.path.join(ROOT, "shared", "inlet_velocity.csv"), delimiter=",", skiprows=1)
        self.inlet_t, self.inlet_u = inl[:, 0], inl[:, 1]

        ref = _load_npz(os.path.join(ROOT, "cases", "case_00", "wall_ref.npz"))
        self.wall_pos0 = ref["wall_pos0"].astype(np.float64)
        self.center = self.wall_pos0.mean(0)
        _, nww = cKDTree(self.wall_pos0).query(self.wall_pos0, k=K_WW + 1)
        self.wall_nbr = nww[:, 1:].astype(np.int64)

        self.snaps = {}
        for c in self.cases:
            self.snaps[c] = {}
            for f in sorted(glob.glob(os.path.join(ROOT, "cases", f"case_{c}", "t_*.npz"))):
                k = int(os.path.basename(f)[2:5])
                z = _load_npz(f)
                assert np.array_equal(z["wall_node_id"], ref["wall_node_id"])
                self.snaps[c][k] = self._prep_snapshot(z)
        self._interp_cache = {}

    # ------------------------------------------------------------------ snapshots
    def _prep_snapshot(self, z):
        pos = z["fluid_pos"].astype(np.float64)
        wall_pos = (self.wall_pos0 + z["wall_disp"]).astype(np.float64)
        _, nf = cKDTree(pos).query(pos, k=K_FF + 1)
        dw, nwf = cKDTree(wall_pos).query(pos, k=K_WF)
        p = z["fluid_p"].astype(np.float64)
        return dict(
            time=float(z["time_s"]), pos=pos, u=z["fluid_vel"].astype(np.float64),
            pbar=float(p.mean()), pprime=p - p.mean(), disp=z["wall_disp"].astype(np.float64),
            wall_pos=wall_pos, fluid_nbr=nf[:, 1:].astype(np.int64),
            wf_nbr=nwf.astype(np.int64), walldist=dw[:, 0])

    def make_snapshot(self, pos, disp, time):
        """Geometry-only snapshot (no fields) for forecasting beyond the data."""
        z = {"fluid_pos": pos, "wall_disp": disp, "time_s": np.array(time),
             "fluid_vel": np.zeros_like(pos), "fluid_p": np.zeros(len(pos))}
        return self._prep_snapshot(z)

    def interp(self, c, ks, kt):
        """Cached IDW weights from on-disk snapshot ks onto snapshot kt's fluid points."""
        key = (c, ks, kt)
        if key not in self._interp_cache:
            self._interp_cache[key] = idw_weights(self.snaps[c][ks]["pos"], self.snaps[c][kt]["pos"])
        return self._interp_cache[key]

    def inlet(self, t):
        return np.interp(t, self.inlet_t, self.inlet_u)

    # ------------------------------------------------------------------ normalisation
    def fit_stats(self, pairs):
        us, pp, pb, dv, wv = [], [], [], [], []
        for c, ks, kt in pairs:
            S, T = self.snaps[c][ks], self.snaps[c][kt]
            us.append(T["u"]); pp.append(T["pprime"]); pb.append(T["pbar"])
            dv.append(T["disp"]); wv.append((T["disp"] - S["disp"]) / ((kt - ks) * DT))
        self.stats = dict(
            u_sd=float(np.concatenate(us).std()),
            pp_sd=float(np.concatenate(pp).std()),
            pbar_mu=float(np.mean(pb)), pbar_sd=float(np.std(pb)),
            disp_sd=float(np.concatenate(dv).std()), wvel_sd=float(np.concatenate(wv).std()))
        return self.stats

    def set_stats(self, stats):
        self.stats = dict(stats)

    def encode_fields(self, u, pprime, pbar):
        s = self.stats
        f = np.concatenate([u / s["u_sd"], pprime[:, None] / s["pp_sd"]], 1)
        return f.astype(np.float32), np.float32((pbar - s["pbar_mu"]) / s["pbar_sd"])

    def decode_fields(self, f, pbar_n):
        s = self.stats
        f = f.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(f) else f
        pbar_n = float(pbar_n)
        u = f[:, :3] * s["u_sd"]
        pprime = f[:, 3] * s["pp_sd"]
        pbar = pbar_n * s["pbar_sd"] + s["pbar_mu"]
        return u, pprime, pbar

    # ------------------------------------------------------------------ samples
    def globals_vec(self, c, t_src, tau_steps):
        t_tgt = t_src + tau_steps * DT
        inl = [self.inlet(t_src)] + [self.inlet(t_tgt - j * 0.005) for j in range(5)]
        inl = np.array(inl) / 0.3
        dudt = (self.inlet(t_tgt) - self.inlet(t_tgt - 0.004)) / 0.004 * (DT / 0.3)
        ph_s, ph_t = 2 * np.pi * t_src / PERIOD, 2 * np.pi * t_tgt / PERIOD
        phase = [np.sin(ph_s), np.cos(ph_s), np.sin(ph_t), np.cos(ph_t)]
        prm = (self.params[c] - self.param_mu) / self.param_sd
        return np.concatenate([inl, [dudt], phase, prm]).astype(np.float32)

    N_GLOBAL = 6 + 1 + 4 + 3

    def sample(self, c, ks, kt, src=None, tgt=None, src_fields=None):
        """Build one transition sample.
        src/tgt: snapshot dicts (default: from disk). src_fields: optional (f_norm (N_src,4), pbar_n) to
        replace the source fields, used during autoregressive rollout."""
        S = src or self.snaps[c][ks]
        T = tgt or self.snaps[c][kt]
        tau = round((T["time"] - S["time"]) / DT)
        idx, w = (self.interp(c, ks, kt) if (src is None and tgt is None)
                  else idw_weights(S["pos"], T["pos"]))
        if src_fields is None:
            f_src, pbar_src = self.encode_fields(S["u"], S["pprime"], S["pbar"])
        else:
            f_src, pbar_src = src_fields
            f_src = f_src.detach().cpu().numpy() if torch.is_tensor(f_src) else f_src
        x_in = idw_apply(idx, w, f_src).astype(np.float32)
        y, pbar_y = self.encode_fields(T["u"], T["pprime"], T["pbar"])

        s = self.stats
        wvel = (T["disp"] - S["disp"]) / (tau * DT)
        wall_x = np.concatenate([
            (self.wall_pos0 - self.center) / L_SCALE, T["disp"] / s["disp_sd"], wvel / s["wvel_sd"],
            np.linalg.norm(T["disp"], axis=1, keepdims=True) / s["disp_sd"]], 1)
        fpos = (T["pos"] - self.center) / L_SCALE
        dev = self.device
        tt = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
        return dict(
            case=c, ks=ks, kt=kt, tau=float(tau),
            x_in=tt(x_in), pbar_in=tt(pbar_src), y=tt(y), pbar_y=tt(pbar_y),
            fpos=tt(fpos), walldist=tt(T["walldist"] / H_SCALE),
            fluid_nbr=tt(T["fluid_nbr"], torch.long), wf_nbr=tt(T["wf_nbr"], torch.long),
            wall_x=tt(wall_x), wall_pos=tt((T["wall_pos"] - self.center) / L_SCALE),
            wall_pos0=tt((self.wall_pos0 - self.center) / L_SCALE),
            wall_nbr=tt(self.wall_nbr, torch.long),
            glob=tt(self.globals_vec(c, S["time"], tau)))


def make_pairs(cases, k_lo, k_hi, taus=(1, 2, 3)):
    """All (case, k_src, k_tgt) with k_lo <= k_src < k_tgt <= k_hi."""
    return [(c, ks, ks + t) for c in cases for t in taus for ks in range(k_lo, k_hi - t + 1)]
