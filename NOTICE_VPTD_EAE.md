# VPTD-EAE implementation boundary

This repository is a standalone VPTD-EAE research implementation owned by
`Chloexf795/VPTD`.

The standalone EAE package does not include or import VAD, `verl`, or EMNLP2026
runtime modules. The optional training launcher uses a separately downloaded,
commit-pinned official Vision-OPD/verl checkout under `external/`, preserving its
upstream license. VAD's temporal counterfactual/attribution idea is related prior work
and must be cited; this repository does not claim that underlying idea as a new
VPTD-EAE invention. EMNLP2026 is a baseline and data-format reference only.

The EAE-specific work represented here includes:

1. temporal attribution over event-role rather than next-token distributions;
2. generated-video consistency gating;
3. support/refutation handling for directional EAE role reversals;
4. leakage-safe SWiG visual-prototype attachment to ACE events;
5. EAE P/R/F1, threshold calibration, and role-reversal diagnostics.

Standalone EAE utilities and tests do not require that external training runtime
or the York-Gold/EMNLP2026 repository. GPU OPD training does require Vision-OPD.
