# AeTHERON-style next-flow prediction (FSI subset)

Preliminary PyTorch implementation of a heterogeneous graph operator following AeTHERON (arXiv:2604.13369),
applied to a 4-case fluid-structure interaction subset. Not a reproduction of the paper's fin results.

Method, results, limitations and commands: [README_AETHERON.md](../README_AETHERON.md)

The dataset is not included. Place `cases/`, `geometry/`, `shared/` and `subset_metadata.json` in the
repository root. Prediction and forecast arrays (`.npz`) are also excluded; regenerate them with the scripts.

Requires Python 3 with numpy, scipy, torch and matplotlib.
