"""AeTHERON-style heterogeneous graph neural operator (arXiv:2604.13369, Sec. 2).

Encoder  : Phi_m([x_m(t+tau), e(tau)]), Phi_f([x_f(t), e(tau)])                     (Eq. 4-5)
Processor: xi_f <- sigma(W xi_f + A_i + A_c), then time-conditioned FiLM             (Eq. 14-15)
           A_i  = mean_j kappa_ff(e_ij) xi_j     (matrix-valued kernel from edge attributes, Eq. 8)
           A_c  = sum_k alpha_ik V_k             (sparse fluid->wall cross attention, Eq. 11-13)
Decoder  : x_f(t+tau) = x_f(t) + tau * psi(xi_f)                                      (Eq. 16)

Graphs are k-nearest-neighbour graphs with fixed in-degree, so every aggregation is a dense
(N, K, ...) tensor op -- no scatter kernels needed (runs on CPU / CUDA / Apple MPS).

Deviations from the paper are documented in README_AETHERON.md: low-rank kernel kappa (kernel width kw), a residual
connection around each processor layer, a short wall-graph processor, and an extra head for the
spatially-uniform pressure level, which is set by the outlet Windkessel.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import H_SCALE, L_SCALE

SPACING = H_SCALE / L_SCALE   # fluid point spacing in model length units (mm)


def mlp(i, h, o, n_hidden=1):
    layers, d = [], i
    for _ in range(n_hidden):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def time_embedding(tau, dim=16):
    """Sinusoidal lead-time embedding (Eq. 3); tau in snapshot units, wavelengths 4..1000 snapshots."""
    k = torch.arange(dim // 2, dtype=torch.float32)
    wl = 4.0 * 250.0 ** (k / (dim // 2 - 1))
    a = 2 * math.pi * tau / wl
    return torch.cat([torch.sin(a), torch.cos(a)])


class WallLayer(nn.Module):
    def __init__(self, dh, de):
        super().__init__()
        self.msg = mlp(2 * dh + de, dh, dh)
        self.upd = mlp(2 * dh, dh, dh)
        self.ln = nn.LayerNorm(dh)

    def forward(self, xs, nbr, e):
        N, K = nbr.shape
        m = self.msg(torch.cat([xs[:, None].expand(N, K, -1), xs[nbr], e], -1)).mean(1)
        return xs + self.ln(self.upd(torch.cat([xs, m], -1)))


class FluidLayer(nn.Module):
    def __init__(self, dh, de_f, de_wf, dt, kw=4, dA=64, heads=4):
        super().__init__()
        self.dh, self.kw, self.dA, self.heads = dh, kw, dA, heads
        self.W = nn.Linear(dh, dh)
        # matrix-valued kernel  kappa_ff(e) = sum_r g_r(e) M_r ,  g: R^de -> R^kw  (kernel width kw)
        # => A_i = sum_r M_r ( mean_j g_r(e_ij) xi_j ): matrices applied after aggregation (cheap)
        self.kernel = mlp(de_f, 32, kw)
        self.M = nn.Linear(kw * dh, dh, bias=False)
        self.q = nn.Linear(dh, dA)
        self.k = nn.Linear(dh, dA)
        self.v = nn.Linear(dh, dA)
        self.ek = mlp(de_wf, 16, dA)
        self.ev = mlp(de_wf, 16, dA)
        self.o = nn.Linear(dA, dh)
        self.ln = nn.LayerNorm(dh)
        self.gamma = mlp(dt, dh, dh)
        self.lam = mlp(dt, dh, dh)

    def forward(self, xf, xs, nbr, e_f, wf, e_wf, temb, tau):
        N, K = nbr.shape
        Kw = wf.shape[1]
        H, dH = self.heads, self.dA // self.heads
        # intra-fluid kernel message passing, A_i  (Eq. 8)
        g = self.kernel(e_f)                                          # (N, K, kw)
        agg = (torch.bmm(g.transpose(1, 2), xf[nbr]) if xf.is_mps else (g.unsqueeze(-1) * xf[nbr].unsqueeze(2)).sum(1)) / K  # (N, kw, dh)
        Ai = self.M(agg.reshape(N, self.kw * self.dh))
        # sparse cross attention fluid (queries) -> wall (keys/values), A_c  (Eq. 11-13)
        Q = self.q(xf).view(N, 1, H, dH)
        Kt = (self.k(xs)[wf] + self.ek(e_wf)).view(N, Kw, H, dH)
        V = (self.v(xs)[wf] + self.ev(e_wf)).view(N, Kw, H, dH)
        att = torch.softmax((Q * Kt).sum(-1) / math.sqrt(dH), dim=1)       # (N, Kw, H)
        Ac = self.o((att[..., None] * V).sum(1).reshape(N, self.dA))
        h = F.silu(self.W(xf) + Ai + Ac)
        h = (1 + tau * self.gamma(temb)) * self.ln(h) + tau * self.lam(temb)  # Eq. 15
        return xf + h


class AeTHERON(nn.Module):
    DE_F, DE_W, DE_WF, D_FIELD, D_WALL = 6, 9, 4, 4, 10

    def __init__(self, n_glob, dh=64, L=8, kw=4, dA=64, heads=4, dt=16, n_wall_layers=2):
        super().__init__()
        self.dt = dt
        self.fenc = mlp(self.D_FIELD + 3 + 1 + 1 + n_glob + dt, dh, dh, 2)
        self.wenc = mlp(self.D_WALL + dt, dh, dh, 2)
        self.wall_layers = nn.ModuleList([WallLayer(dh, self.DE_W) for _ in range(n_wall_layers)])
        self.layers = nn.ModuleList([FluidLayer(dh, self.DE_F, self.DE_WF, dt, kw, dA, heads)
                                     for _ in range(L)])
        self.dec = mlp(dh, dh, self.D_FIELD, 2)
        self.pbar_head = mlp(dh + n_glob + 1 + dt, dh, 1, 2)

    @staticmethod
    def edge_features(b):
        fp, nbr = b["fpos"], b["fluid_nbr"]
        rel = (fp[nbr] - fp[:, None]) / SPACING
        wd = b["walldist"]
        e_f = torch.cat([rel, rel.norm(dim=-1, keepdim=True),
                         wd[:, None, None].expand_as(rel[..., :1]), wd[nbr][..., None]], -1)
        wp, wp0, wn = b["wall_pos"], b["wall_pos0"], b["wall_nbr"]
        r = (wp[wn] - wp[:, None]) / SPACING
        r0 = (wp0[wn] - wp0[:, None]) / SPACING
        d, d0 = r.norm(dim=-1, keepdim=True), r0.norm(dim=-1, keepdim=True)
        e_w = torch.cat([r, d, r0, d0, 10.0 * (d / d0.clamp_min(1e-6) - 1)], -1)
        rwf = (fp[:, None] - wp[b["wf_nbr"]]) / SPACING
        e_wf = torch.cat([rwf, rwf.norm(dim=-1, keepdim=True)], -1)
        return e_f, e_w, e_wf

    def forward(self, b, x_in=None, pbar_in=None):
        x_in = b["x_in"] if x_in is None else x_in
        pbar_in = b["pbar_in"] if pbar_in is None else pbar_in
        dev = x_in.device
        tau = b["tau"]
        temb = time_embedding(tau, self.dt).to(dev)
        N = x_in.shape[0]
        e_f, e_w, e_wf = self.edge_features(b)
        g = b["glob"]
        xf = self.fenc(torch.cat([x_in, b["fpos"] / 5.0, b["walldist"][:, None] / 4.0,
                                  pbar_in.expand(N, 1), g.expand(N, -1), temb.expand(N, -1)], -1))
        Nw = b["wall_x"].shape[0]
        xs = self.wenc(torch.cat([b["wall_x"], temb.expand(Nw, -1)], -1))
        for lyr in self.wall_layers:
            xs = lyr(xs, b["wall_nbr"], e_w)
        for lyr in self.layers:
            xf = lyr(xf, xs, b["fluid_nbr"], e_f, b["wf_nbr"], e_wf, temb, tau)
        f_out = x_in + tau * self.dec(xf)
        pbar_out = pbar_in + tau * self.pbar_head(
            torch.cat([xf.mean(0), g, pbar_in.view(1), temb])).squeeze()
        return f_out, pbar_out
