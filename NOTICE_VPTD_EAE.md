# VPTD-EAE implementation boundary

This repository is a standalone VPTD-EAE research implementation owned by
`Chloexf795/VPTD`.

It does not include or import the VAD repository, `verl`, or EMNLP2026 runtime
modules. VAD's temporal counterfactual/attribution idea is related prior work
and must be cited; this repository does not claim that underlying idea as a new
VPTD-EAE invention. EMNLP2026 is a baseline and data-format reference only.

The EAE-specific work represented here includes:

1. temporal attribution over event-role rather than next-token distributions;
2. generated-video consistency gating;
3. support/refutation handling for directional EAE role reversals;
4. leakage-safe SWiG visual-prototype attachment to ACE events;
5. EAE P/R/F1, threshold calibration, and role-reversal diagnostics.

No source file from VAD/verl or York-Gold/EMNLP2026 is required at runtime.
