# Verification-01 — PI check of selements' revised §IV paragraph

**Consumer:** selements-website · **Follow-up to:** `tomography-framing-check` (this folder) · **Responder:** CosmoGasVision PI · **Branch:** `service/data-export` · **Verb-ceiling:** [D-37] + [D-73] (binding). Prose advice; critique not rewrite.

Selements folded the prior ruling's three fixes into a revised ep01 §IV paragraph and asked the PI to verify legitimacy / catch residual overclaim, sugarcoat, overframing.

## Verdict: APPROVE WITH TWO BINDING CORRECTIONS

Fixes 1 and 2 are clean; fix 3 is half-done (the "z≈2–3 side" named, the "CGV side" broken). Two clauses are binding; everything else clears.

**Binding arithmetic (re-verified this session):** CGV synthetic mean sightline separation = 60 h⁻¹Mpc / √1024 = 60/32 = **1.875 ≈ 1.9 h⁻¹Mpc** (uniform-grid and area-per-ray both give 1.875) — **denser than** CLAMATO's 2.37 h⁻¹Mpc, not sparser. This is what makes correction 1 binding.

### Correction 1 (BINDING) — the final sentence misreads as "we reproduced sparse z=0.3 conditions"
Draft: *"Our experiment isolates the faint-absorption regime: synthetic sightlines through a simulated box at z = 0.3, asking whether the 3D gas can be recovered there at all."* Placed right after "we do not have that dense grid of close-packed backlights," a lay reader parses "isolates the faint-absorption regime" as *reproduced the sparse z=0.3 geometry.* False — CGV holds sampling **dense** (~1.9 h⁻¹Mpc) and varies **only** absorption.

Corrected (binding):
> "Our experiment isolates the faint absorption itself: we keep the sightlines densely packed — as in the z ≈ 2–3 surveys, not as sparse as a real z = 0.3 survey would be — and change only the strength of the absorption, then ask whether the 3D gas can be recovered from that alone."

Shorter binding-compliant form:
> "Our experiment isolates the faint absorption alone: the synthetic sightlines are kept densely packed — denser than a real z = 0.3 survey — so that only the absorption strength changes, and we ask whether the 3D gas can be recovered there at all."

### Correction 2 (BINDING TIGHTEN) — "improve on that reconstruction" implies TARDIS improved the real map
Draft: *"later methods built for the same problem improve on that reconstruction, for example TARDIS… so far demonstrated on simulated data."* "that reconstruction" = the real CLAMATO map; TARDIS Paper I improved on Wiener *in simulation*, not on the real map. The trailing hedge mitigates only after the misread lands.

Corrected (binding):
> "…and later methods aim to improve on that kind of reconstruction — for example TARDIS (Horowitz et al. 2019), a forward-modeling approach that improves on Wiener filtering in simulations and so far has been demonstrated only on simulated data."

Minimal-edit alternative: change "improve on that reconstruction" → "improve on that approach in simulations."

### Cleared (no change needed)
- **"demonstrably works"** attaches only to the CLAMATO/Wiener real-data clause; never touches CGV. Clean.
- **Closing question** ("whether the 3D gas can be recovered there at all") — in-scope for a primer; a question, no verdict, no false suspense, no sugarcoat. Clean. *(Note: the revised paragraph folded the close into the last sentence and dropped the standalone "Given only the sparse 1D flux, can a model reconstruct the 3D gas behind it?" line the prior ruling endorsed keeping. Storytelling choice — not binding — but restore it if it was dropped by accident.)*
- **Physics:** "gas is denser / absorption far stronger" at z≈2–3 vs z=0.3 — correct. "a couple of megaparsecs" for 2.37 h⁻¹Mpc — fair lay rounding (and doubly fair now, since CGV's ~1.9 h⁻¹Mpc is also "a couple of megaparsecs").

## Summary
Two binding clause edits (final sentence + "improve on"). With those landed, the paragraph is APPROVED and within the [D-37] + [D-73] verb-ceiling. Storytelling around them stays the editor's.

Sources verified first-hand: CLAMATO Lee+2018 (arXiv:1710.02894); TARDIS Horowitz+2019 (arXiv:1903.09049). Grounded in `experiments/nerf/design/D73_project_diagnosis_v2.md` §1/§4/§6.
