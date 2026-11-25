# AeTHERON-style next-flow prediction on the FSI student subset

This is an implementation of the heterogeneous graph neural operator from *AeTHERON* (arXiv:2604.13369). It is trained on the 4-case subset and predicts the fluid state (velocity and pressure) at the next snapshot.

## What the data is

The subset holds 4 compliant-vessel FSI cases (00, 06, 13, 20). Each has 31 snapshots at t = 0.90–1.50 s, spaced Δt = 0.02 s apart. A snapshot contains about 4000 fluid points (velocity and pressure) and 4246 wall nodes (displacement).

- **Flow driver:** a shared inlet waveform with a period of exactly 0.30 s (15 snapshots), so the data covers 2 cycles.
- **Fluid points change every snapshot.** The solver remeshes, so fluid node indices are not persistent across snapshots.
- **Wall rows do persist** across snapshots.
- **Case parameters:** μ, α and s. The Windkessel values R/C/Z are fixed multiples of s.

## Method (`aetheron/`)

| File | Contents |
|---|---|
| `data.py` | Builds one heterogeneous graph pair per transition t → t+τ (τ = 1, 2 or 3 snapshots) |
| `model.py` | Encoder, then 8 processor layers, then a residual decoder (0.50M parameters) |
| `train.py` | Splits `temporal`, `unseen_case` and `all` |
| `evaluate.py` | One-step and rollout metrics against baselines, plus figures |
| `forecast.py` | Rollout beyond the data (t = 1.52–1.80 s) |

How the paper's design maps onto this data:

- **Fluid graph.** The nodes are the target snapshot's fluid points; only their geometry is used. The source fields at time t are moved onto these points by inverse-distance interpolation (4 neighbours). This handles the remeshing. Edges link each point to its 8 nearest neighbours.
- **Wall graph.** Each wall node has 10 features: reference position, displacement at t+τ, wall velocity and |displacement|. As in the paper (Eq. 4), the wall state at the lead time is prescribed.
- **Wall → fluid coupling.** Each fluid node attends to its 8 nearest deformed wall nodes through sparse cross-attention (Eqs. 11–13).
- **Processor update.** Each layer computes ξ ← ξ + FiLM_τ(LN(σ(Wξ + A_i + A_c))), where A_i comes from a matrix-valued kernel κ(e) = Σ_r g_r(e)·M_r with kernel width 4 (Eq. 8). FiLM_τ is the time conditioning of Eq. 15.
- **Decoder.** x(t+τ) = x(t) + τ·ψ(ξ), as in Eq. 16. A separate pooled head predicts the domain-mean pressure, which the outlet Windkessel sets. The spatial pressure fluctuation p′ is predicted per node.
- **Global conditioning.** The inlet waveform over [t, t+τ], the cardiac phase, (μ, α, s) and the sinusoidal τ embedding.
- **Training.** Adam at lr 2e-3, halved every 10 epochs, for 30 epochs. The loss is L1 on normalized u, p′ and mean pressure. Gaussian input noise (σ = 0.02) is added for rollout robustness.

### Differences from the paper

- The paper's caudal-fin data isn't in this subset. The vessel cases here are a different FSI problem.
- The kernel is low-rank, and each processor layer has a residual connection.
- The model predicts pressure as well as velocity.
- Inlet-waveform and phase conditioning are added, because flow in this vessel is driven by the inlet rather than by body motion.
- Training ran for 30 epochs instead of 200, and the latent dimension is d_h = 64.

## Results

Velocity MAE is in m/s and pressure MAE in Pa. Metrics are averaged over the test snapshots.

### A) Temporal extrapolation (36 test snapshots: 4 cases x t = 1.34-1.50 s)

Train on t <= 1.32 s. The test phases were all seen one cycle earlier in training, so this is not a
strong generalization test.

| Prediction | vel MAE | vel rel-L2 | p MAE | spatial p' MAE |
|---|---|---|---|---|
| AeTHERON, one-step | 0.0042 | 0.272 | **21.4** | 2.10 |
| Persistence, one-step | 0.0116 | 0.549 | 290.5 | 4.94 |
| Previous cycle (lookup) | **0.0036** | **0.232** | 29.0 | **0.34** |
| AeTHERON, 9-step rollout | 0.0135 | 1.707 | 84.6 | 4.12 |
| Frozen start state | 0.0835 | 11.35 | 1131 | 25.9 |

### B) Unseen case (case 20 held out, 30 snapshots)

| Prediction | vel MAE | vel rel-L2 | p MAE | spatial p' MAE |
|---|---|---|---|---|
| AeTHERON, one-step | **0.0055** | **0.210** | **60.1** | 2.20 |
| Persistence, one-step | 0.0170 | 0.524 | 391.0 | 7.27 |
| Mean of cases 00/06/13 at same time | 0.0075 | 0.355 | 202.3 | **1.19** |
| AeTHERON, 30-step rollout | 0.0107 | 0.712 | 294.5 | 2.44 |
| Frozen start state | 0.0340 | 0.732 | 1052 | 10.3 |

On the 16 snapshots that have a previous cycle: AeTHERON one-step 0.0053 / 0.215 / 63.2 / 2.16 vs
previous cycle 0.0054 / 0.243 / 42.4 / 0.83 (same columns).

### What the results show

- **Supported:** one-step velocity on the unseen case beats persistence (29/30 snapshots) and the
  other-cases mean (26/30). Against the previous cycle it is a tie (8/16 by MAE).
- **Not supported:** long rollouts. They beat a frozen start state, but lose to the same lookups
  (A: 0.0135 vs 0.0036; B: 0.0107 vs 0.0075). Rollout rel-L2 in A is 1.7, worse than predicting zero flow.
- **Spatial pressure** is poor: p' error is 2-6x the lookups. Good total pressure comes from the mean level.
- **Extra information:** the model receives the true future wall motion and fluid geometry (prescribed, as
  in the paper). Wall displacement tracks pressure, so this may explain part of the pressure accuracy.

## Forecast beyond the data (`outputs/forecast/`)

The model rolls out from t = 1.50 s through one more cycle (15 steps, t = 1.52–1.80 s). There is no ground truth for this window. Two inputs don't exist past 1.50 s and are taken from one period earlier:

- **Fluid query points:** reused from t − 0.30 s.
- **Prescribed wall displacement:** taken from t − 0.30 s, plus the observed cycle-to-cycle drift.

The forecast used the `temporal` model.

What the forecast reproduces:

- the systolic peak at t = 1.60 s, with peak mean speed of 0.23–0.25 m/s against 0.22–0.24 m/s in the previous cycle;
- the timing and case-specific level of the mean-pressure pulse.

The cycle forecast differs from the previous cycle by mean rel-L2 of 0.92-1.35 (no ground truth exists), so its spatial detail should not be trusted. The late-diastole tail of the cycle forecast overestimates the residual flow, at about 0.03 m/s mean speed against about 0.008 m/s.

Files are `case_XX_t_0YY.npz`, containing fluid_pos, fluid_vel, fluid_p, time_s and wall_disp.

## Reproduce

```bash
python3 -m aetheron.train --split temporal --device cpu --epochs 30
python3 -m aetheron.train --split unseen_case --device mps --epochs 30
python3 -m aetheron.evaluate --split temporal
python3 -m aetheron.evaluate --split unseen_case
python3 -m aetheron.forecast --model outputs/temporal/model.pt
```

## Caveats

- **Small, coarse data.** There are only 4 cases, and the point clouds are voxel-subsampled to about 4000 points. The README of the subset says these clouds are not for solver-grade conclusions.
- **Interpolation error is included.** The interpolation step that handles remeshing adds error, and that error is part of every number above, including the baselines.
- **Single runs.** Each result comes from one training run with seed 0. Run-to-run variance wasn't measured.
- **Small preprocessing leak.** The scaling of (mu, alpha, s) uses all 4 cases, including held-out case 20.
  No case 20 flow data was used in training for split B. Not yet fixed.
- **Rollout plots.** The rollout panels in `error_vs_time.png` show only the frozen-state baseline; use the tables above.
