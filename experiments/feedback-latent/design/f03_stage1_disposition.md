# [F-03] Stage-1 smoke disposition (PI, 2026-07-24)

> **VERDICT: F0 fix ADMISSIBLE; v2 (lr=1e-4) is the Stage-1 read. Plumbing PASS-with-scoped-finding (signed off). Latent-separation verdict BLOCKED-pending-panel — the whole-field c1/c2 gates are mis-specified for a shared-IC small-effect suite (now PROVEN, §3). NOT F-E (c4 forbids: codes separate 5.48× above the label-shuffle null). The whole feedback-latent science claim now rests on R-B clearing its dual-null on the DIFFERENCE field.**

Inputs verified line-by-line against: `stage1_lr_audit.json`, `stage1_smoke_results_v1_lr1e-3_RED.json`, `stage1_smoke_results.json` (v2), `true_field_headroom.json`, `f02_founding_ratification.md`.

## 1. F0 status — admissible, with a mandatory spec-text amendment
Gate BARS are byte-identical v1↔v2 (gate_i 0.28587, gate_ii 0.05661, gate_iii 0.15881; all derived from `mse_mean_floor=0.31763` and `target_std=0.56609` — data properties, not lr). The F0 audit diagnosed a real SIREN(ω₀=30) optimization pathology: lr=1e-3 stuck at the mean-floor (`min_over_floor=0.9985`), loss non-monotonic up; lr∈{1e-4,3e-4} overfit a fixed batch to ~6e-4/2e-4. lr is an optimizer hyperparameter; changing it restores convergence without touching what counts as PASS. **Run-config correction, not gate-bending. v2 is the admissible read.**
**Amendment of record (R26, avoid inherited-stale-spec):** `f02` §4(c) s2 literally wrote "lr 1e-3"; that text is stale. Corrected to lr=1e-4 with `stage1_lr_audit.json` as the derivation.

## 2. Central call — NOT F-E; small-effect is the live reading; smoke cannot certify F-A vs F-B
c4 is decisive: conditioned pairwise code L2 = 0.0716 vs label-shuffle null 0.0131 = **5.48× above null**. The codes did not collapse — the F-E gloss ("z_p carry no variant information") is **falsified**. So the literal c2-fail and the F-E definition have come apart; this is NOT the vacuous-latent cell.
- Smoke CAN conclude: F-E-strict (collapsed/decorative latent) ruled out.
- Smoke CANNOT conclude: whether the non-collapsed codes decode the TRUE feedback response (F-A) or separated on a nuisance direction (F-B, method-not-science). c4 shows the codes are *different*, not that they are *correct*. Only R-B (decoded D_i vs TRUE D_i above the measured null) discriminates F-A from F-B.

## 3. Gate-construction — c1/c2 whole-field r_s(σ=2) ARE mis-specified. PROVEN.
Required proof-of-mis-specification measurement (`true_field_headroom.json`): whole-field r_s(σ=2) between TRUE P_i and TRUE P_1:

| pair | r_s(σ=2) whole-field | r_s(σ=1) | std(D_i)/std(field) |
|---|---|---|---|
| P2 (stellarwind) vs P1 | 0.99947 | 0.99918 | 0.084 |
| P3 (windAGN) vs P1 | 0.99738 | 0.99429 | 0.239 |
| P4 (windstrongAGN) vs P1 | 0.96957 | 0.93597 | 0.681 |

The variants' whole fields are **0.97–0.9995 correlated**. A whole-field swap-test therefore had headroom of at most ≈0.03 (P4) down to ≈0.0005 (P2) — against a 0.0565 bar. **c1/c2 could not have passed even with a perfect latent.** The gate had no discriminating headroom by construction; it conflated (A) vacuous latent [falsified by c4] with (B) real latent below the whole-field probe's resolution. Mis-specification is demonstrated, not argued.
**Corollary (the real signal):** the difference field D_i = x(P_i) − x(P_1) has real, *ordered* amplitude — P4 std(D)=0.369 (68% of field std), P3 0.130, P2 0.045. This is the feedback ladder. The feedback signal is real and lives in the difference, exactly where R-B looks; P4−P1 is the strongest-SNR pair.
**Correct probe:** swap-test / separation on D_i at the σ where feedback lives (σ∈{1,2}; the σ choice needs its own derivation). **This is a pre-registered gate correction → REQUIRES the B6 panel micro-cycle (Ext-2 rule 6) BEFORE any re-run. Do NOT silently re-run a redefined gate.**

## 4. λ_z pin is provisional
The sweep's internal c1_separation (0.0654 at λ_z=0) is NOT reproduced by the authoritative controls c1 (0.00532): the sweep's shared baseline (Q_shared=0.460) under-converged while the properly-converged control shared baseline (0.544) caught up to conditioned (0.550). The separation the pin rests on evaporates under the fair control. **λ_z=0 is provisional; re-derive it on the difference-field probe.**

## 5. Stage-1 exit disposition — split
- **Plumbing: PASS (signed off).** s2 all three sub-gates PASS (0.1414<0.2859; 0.348>0.0566; 0.00114<0.1588); s3 contract PASS; architecture fits structure (whole-field r_s≈0.52–0.55); codes non-collapsed (c4). Legitimate Stage-1 deliverable.
- **Anti-collapse latent-separation: DEFERRED** — not PASS (c1+c2 separation not satisfied as written), not RED (c4 forbids F-E). Honest state = INCONCLUSIVE-by-mis-specification; verdict deferred to the difference-field probe behind B6.
- **Honest paper implication ([D-37], symmetric):** the whole-field near-degeneracy IS the Nasir+2017 small-effect signature surfacing already at smoke. It raises F-B probability and narrows the science margin. The entire feedback-latent science claim now rests on **R-B clearing its dual-null on the difference field.** This does not establish F-B — c4 keeps F-A and F-B both live — but it is disclosed as an early warning, not buried. **The whole ballgame is R-B.**

## 6. Next step + owed gates (all $0/local; NO Juno; nothing re-runs a gate)
Mandatory next action: **B6 panel micro-cycle scoped to the gate correction** (c1'/c2' on D_i). Prerequisites to feed the panel:
- **B3 (G0 on the difference-field path): DISCHARGED** — `b3_g0_acceptance.json`: D=x(P2)−x(P1) self-corr=1.0 at σ{1,2,4}; phase-randomized null N=200 σ2 97.5pct=0.118; constant→UNDEFINED. (PI R15 flag #5 noted it unrun; it is in fact run — this closes it.)
- **True-field headroom: DISCHARGED** — `true_field_headroom.json` (§3).
B6 panel-pre-review scope for c1'/c2': (a) object = D_i = x(P_i)−x(P_1); (b) σ choice with derivation; (c) separation bar DERIVED from a measured null; **note c1'/c2' on D_i is nearly identical to R-B's N1 mismatched-difference null — the panel must rule whether c1'/c2' is a distinct smoke gate or is subsumed directly into R-B**; (d) amplitude/SNR disclosure — if std(D_i^true) near the G0(b) estimator floor → UNDEFINED (F-F), not FAIL (relevant to P2, std(D)=0.045, though that is ≫ the 1e-12 floor and ≈ the phase-randomized null 0.118 → P2 is the low-SNR pair to watch).
R28/R29 on c1'/c2': bars derived-at-spec-time from measured quantities (difference-field seed-SD or the R-B null band), never chosen; unit-chain D_i → r_s(σ) written+discharged at gate-construction commit; observable in the same σ/frame as R-B's science verdict.

## R15 — PI reliance flags this dispatch
1. Did not re-run `nccf.py`; r_s/Q values as-reported by the smoke driver, not independently recomputed. (Coordinator note: the §3 headroom numbers WERE computed directly via nccf this turn.)
2. Did not re-read the smoke driver to confirm c1/c2/c4 compute what their labels claim.
3. Did not view the four `stage1_slab_P*.png` figures; §4(e) eyeball step NOT discharged — owed before the exit artifact closes.
4. c3 `"degrades": true` is numerically trivial (Δ=7.5e-5); diagnostic-only, no magnitude bar — a wart, does not gate.
5. B1/B2/B4 statuses taken from spec, not re-verified. (B3-on-differences now confirmed discharged, §6.)
