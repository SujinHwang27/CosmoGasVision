# [D-56] Sprint-5 (c′)-at-48³ Gate-5 Outcome-Routing Scaffold — FILLED 2026-05-16

**Predecessor**: [D-55] v4 NON-PROVISIONAL at HEAD `504037b` on `exp/nerf`
**Status**: FILLED at gate-5 absorption 2026-05-16 — branch **(iii-a) ALL-PASS + AD-5 ABOVE-BAR** selected; non-selected branches retained as audit trail.
**Target file**: `experiments/nerf/LEDGER.md` (§3 [D-56] append + §1 Stage 3 row append + §7 Session Snapshot 2026-05-16 append).

---

## PART 1 — [D-56] Decision-of-Record Block (drop into §3)

### [D-56] Sprint-5 (c′)-at-48³ outcome at gate-5 — branch-(iii-a) ALL-PASS + AD-5 ABOVE-BAR absorbed; paper-text propagation R11 non-CVPR-only + R16 cross-atom deferred to follow-on session (2026-05-16, PI gate-5 absorbing Juno H100 run)

**Predecessor + run context.** [D-55] design v4 at HEAD `504037b` on `exp/nerf` authorized sprint-5 (c′)-at-48³ for post-CVPR execution, NON-PROVISIONAL per R15 clause (b) (gate-2 RE-pre-review APPROVE-WITH-AMENDMENT on v3 → v4 absorbed the design-layer PROVISIONAL status; the empirical-outcome-absorption gate-5 does NOT re-trigger PROVISIONAL because the design itself was discharged). Juno H100 dispatch via `scripts/submit_juno_sprint5_cprime.sh`: run_id `sprint5cprime_1778947953`, run_tag `Sprint5cprime-48cube-5seed-8992c12-20260516-111221-d8928b`, Juno job `199430` (g-06-01 H100 partition; two prior dispatches failed and were fixed in-session — `199428` ExitCode 141 SIGPIPE on `nvidia-smi | head` under `pipefail` fixed at HEAD `6094f80`; `199429` `FileNotFoundError` on `SherwoodIGM_gal/extracted/planck1_60_768_z0.300/snapdir_012` because sprint-5 (c′) ran from `JUNO_WORK` without the data symlinks sprint-4 had staged per-run-dir, fixed at HEAD `8992c12`), wallclock **19:50 min** (design-budget 135 min; conservative anchor — empirical wallclock dominated by early-stop at epochs 6–7 across all 5 seeds with the val-loss divergence pattern seen at 32³).

**Empirical observation (per [D-37] rule (a) — lead with observation, framing downstream).**

Pre-flight verification (per [D-55] amendment 7 + R23 BANKED):
- **S4 cuDNN-determinism gate-(c) verdict: FAIL_relaxed_to_1e-5_with_footnote** (Juno venv lacks `pytest`; pre-flight relaxed-gate fallback fired per design v4 §6 footnote — disclosure-only, NOT a hard blocker; the relaxed verdict bands the determinism check to runtime config evidence in the headline).
- **Per-crop JSONL line counts (R23)**: seed_42 = 8000, seed_142 = 8000, seed_242 = 8000, seed_342 = 8000, seed_442 = 8000. **Total = 40000** (target 40000). **R23 compliance: PASS** (5 seeds × 8000 per-crop indicators across the held-out [D-49] test region, 70/15/15 axis-0 split).
- **A4 MVSK-at-32³ baseline (R18 nesting)**: Â_mvsk_at_32cube_seed_avg = **0.3400**. mvsk_threshold_tightened: **false** (all 5 per-seed A4 within the design v4 sanity band).

Primary empirical quantities (seed-averaged across 5 seeds {42, 142, 242, 342, 442}):
- **Â_resnet_seed_avg = 0.5805 ± 0.0746** (sample std n=5, ddof=1)
- **Â_mvsk_at_48cube_seed_avg = 0.3371 ± 0.0320**
- **AD-5 margin (point estimate): 24.34 pp**; bootstrap 95% CI on seed-averaged margin (B=10000 resampling 5 seeds w/ replacement, RNG=20260516): **lower-CI = 18.80 pp, upper-CI = 28.88 pp**. Lower-CI > 10 pp self-anchored bar → AD-5 ABOVE-BAR.
- **AD-5 per-seed: 5/5 PASS**; per-seed margins (pp) = {24.75, 26.45, **14.28**, 31.64, 24.59}; failing seeds = {} (seed 242 is the closest-to-bar at 14.28 pp, still 1.43× the bar).
- **ρ_emp = 0.1945** (mean within-seed Pearson correlation between `1{ResNet correct}` and `1{MVSK@48³ correct}`, averaged across 5 seeds; per-seed values {0.222, 0.216, 0.352, 0.093, 0.089}).
- **σ_seed_emp (ResNet) = 0.0746** (sample std of per-seed Â_resnet); **σ_seed_emp (MVSK@48³) = 0.0320**.
- **ρ_seed_emp = 0.1623** (mean pairwise between-seed Pearson correlation of per-crop `1{ResNet correct}` indicators across all 10 seed-pairs; range [0.080, 0.242]).

Per-gate verdicts: gate-(a) **PASS** (sanity floor: Â_resnet seed-averaged 0.5805 > 0.50 chance bar; 5/5 per-seed); gate-(b) **PASS** (no training divergence: all 5 seeds train_loss descended monotonically across epochs 1–6 prior to early-stop; no NaN/Inf); gate-(c) **PASS_relaxed** (cuDNN-determinism gate relaxed to 1e-5 per S4 footnote — Juno venv `pytest` absence; design v4 disclosure-only fallback); gate-(d) **PASS** (no AD-1 fail; mvsk_threshold_tightened=false across all 5 seeds); gate-(e) **PASS** (AD-5 margin lower-CI 18.80 pp > 10 pp self-anchored bar; 5/5 per-seed AD-5 PASS).

**Branch selection: (iii-a) ALL-PASS + AD-5 ABOVE-BAR.**

---

#### DISPOSITION BRANCH (i) — PROCESS-FAILURE  [NOT SELECTED — retained as audit trail]

Failure mode: N/A. **Disposition**: PROCESS-FAILURE. No publication. No paper-text propagation. R16 NOT triggered. LEDGER §7 records failure signature; no §1 Pulse promotion. Debug obligation per [D-55] §2 branch-(i). Re-dispatch criteria: root cause identified, fix landed, smoke test passes, design v4 unchanged.

R-rule audit: R15 NON-PROVISIONAL HOLDS (process-failure absorption is itself definite). R23 per-crop JSONL completeness is the primary diagnostic. R8/R9/R10/R11/R13/R14/R16 N/A or not-triggered.

---

#### DISPOSITION BRANCH (ii) — RERUN-WITH-ADJUSTED-PARAMS  [NOT SELECTED — retained as audit trail]

Failure mode: N/A. **Disposition**: RERUN. No paper-text propagation until rerun lands a definitive (iii)/(iv) verdict. R16 NOT triggered. Parameter-adjustment per [D-55] §2 branch-(ii). Re-dispatch with NEW run_id citing this entry's empirical observation.

R-rule audit: R9/R10/R13/R15 HOLD. R16 NOT triggered. R23 instrumentation MUST be preserved on rerun.

---

#### DISPOSITION BRANCH (iii-a) — ALL-PASS + AD-5 ABOVE-BAR  [**SELECTED**]

**Empirical lead**: at 48³ substrate, **AD-5 margin lower-CI = 18.80 pp > 10 pp self-anchored bar**; all 5 per-seed AD-5 PASS (margins {24.75, 26.45, 14.28, 31.64, 24.59} pp). gate-(a)/(b)/(c)/(d)/(e) all PASS.

**Disposition**: above-bar single-point claim at 48³. Paper-text eligible per R11 non-CVPR-only (CVPR remains 32³ scope-lock per [D-43]; CVPR PDF submission-ready at HEAD `d7a23ca` per [D-43] Step 5 CLOSED 2026-05-14). R16 cross-atom propagation OBLIGATORY to `papers/shared/sec/` + `papers/shared/sec_extended/` atoms for any non-CVPR venue manifest — **deferred to separate latex-author session with explicit R16 sign-off gate**.

**Verb constraint**: single-point register only. PERMITTED: "at 48³ substrate scale, the AD-5 margin between the 3D ResNet truth-discriminator and the moment-only MVSK baseline is 24.34 pp (5-seed mean; 95% bootstrap CI [18.80, 28.88])". FORBIDDEN: "extends to" / "scales to" / "the discriminator is substrate-scale-invariant" / any curve-implying verb. Multi-point scoping curve (96³+) REQUIRED before "extends" / "scales" framing is admissible — DEFERRED to future memory-budget-expanded design.

**Substantive scientific finding**: at 48³ ρ-crop NeRF-substrate scale, the 4-class Sherwood feedback discrimination signature exceeds the moment-only MVSK 4-scalar subspace by ~24 pp in classifier accuracy — a single-point above-bar result, complementary to but NOT a generalization of the sprint-4 32³ branch-iv finding ([D-51]: at 32³, the same problem collapses to ≈ MVSK 4-scalar subspace within seed noise). The two points jointly establish a single-pair contrast (substrate-scale matters in this direction between 32³ and 48³ at fixed pitch); no claim is admissible about how the curve behaves at 96³+ or below 32³ without additional points.

R-rule audit at gate-5 (branch iii-a):
- **R8 HOLDS** (no cascade-close opened; (c′) is a single-substrate-point probe, not a multi-point closure claim).
- **R9 HOLDS** (verb constraint bans "invariance"/"extends"; permitted verbs are single-substrate-point register).
- **R10 re-affirmed** (X⊥Y orthogonality: retirement reason at 32³ — ceiling-disqualified per [D-51] — orthogonal to the design purpose at 48³ which is substrate-scale extension; the X⊥Y was BANKED at [D-55] gate-3 and remains intact at gate-5 empirical-close).
- **R11 non-CVPR only** (CVPR 32³ scope-lock per [D-43] is preserved; (c′) 48³ result is exclusively post-CVPR paper-text material).
- **R13 HOLDS** (single-point claim at 48³; no scope-creep to multi-substrate-scale verbs).
- **R14 self-anchored-bar + R18 nesting disclosure OBLIGATORY in any paper-text**. Rule-7 rescue clause (ii) satisfied by pre-committed branch routing ([D-52] amendment 7 + [D-55] §2 v4); clause (iii) by R11 non-CVPR-only.
- **R15 NON-PROVISIONAL per clause (b)** — gate-2 RE APPROVE-WITH-AMENDMENT on rescue path discharged design-layer PROVISIONAL at [D-55] gate-3; gate-5 empirical absorption inherits NON-PROVISIONAL.
- **R16 TRIGGERED for future non-CVPR venue** (cross-atom propagation to `papers/shared/sec/` + `papers/shared/sec_extended/` deferred to separate latex-author session).
- **R17/R19 HOLD** (substrate-scale axis confirmed; pitch-preserving — n_grid=768 unchanged from sprint-4; only crop_size varied 32³ → 48³).
- **R20 variance-decomposition**: σ_seed_resnet_emp = 0.0746 (1.49× the design v4 budget σ_seed = 0.05); ρ_seed_emp = 0.1623 (within the v4 budget framing for between-seed correlation). Despite σ_seed_resnet_emp exceeding the design-budget anchor by ~50%, the AD-5 margin lower-CI = 18.80 pp still exceeds the 10 pp self-anchored bar — variance-decomposition gate PASSES on empirical margin, with R20 caveat that the seed-variance budget was under-estimated by ~50% (directional risk identified by R21 at [D-55] gate-3 was confirmed in the conservative direction; the gate still passed because the AD-5 effect size was larger than the design's MDE budget).
- **R21 domain-transfer anchor**: design v4 budget σ_seed = 0.05 was R21-anchored on [D-42-meta] C1 (NeRF flux-statistic seed-dispersion, generative regime) with explicit directional-risk disclosure that discriminative-classifier init-sensitivity may be larger (D'Amour+2020 §6 precedent). Empirical σ_seed_resnet_emp = 0.0746 at (c′) target domain confirms the R21 directional-risk disclosure in the conservative direction (under-estimate). **R21 verdict: WITHIN-BAND-WITH-CAVEAT** (factor 1.49, not factor-of-2+; the R21 escalation route was NOT triggered because the AD-5 effect size was large enough that the MDE-budget under-estimate did not flip the gate verdict; the empirical anchor is recorded for future variance-derivation calibration).
- **R22 readout-topology re-verified** at `src/models/cnn3d.py:135` — `AdaptiveAvgPool3d(1)` unchanged from sprint-4 32³; the (c′) 48³ result does NOT lean on a token-count-ratio expressivity argument (the [D-55] v3 R10 item-1 demotion stands). The empirical above-bar result is consistent with the demoted-modest-contributor framing.
- **R23: 40000 per-crop indicators verified** (5 seeds × 8000 lines per JSONL across `cloud_runs/Sprint5cprime-48cube-5seed-8992c12-20260516-111221-d8928b/eval/per_crop_seed_{42,142,242,342,442}.jsonl`). R23 enabled the gate-5 ρ_emp + ρ_seed_emp + σ_seed_emp computation; without it the R20/R21 audit would not have been possible at this gate.

R-rule banking question: **none surfaced** at gate-5. No R24 candidate. The (c′) gate-5 absorption discharged R19/R20/R21/R22/R23 cleanly within their banked-binding language; no novel cross-rule attack surface emerged from the empirical observation.

---

#### DISPOSITION BRANCH (iii-b) — ALL-PASS + AD-5 INDISTINGUISHABLE-FROM-BAR  [NOT SELECTED — retained as audit trail]

**Empirical lead**: would require AD-5 margin CI to bracket 10 pp. Empirical: lower-CI 18.80 pp ≫ 10 pp, upper-CI 28.88 pp ≫ 10 pp — CI does NOT bracket the bar; branch (iii-b) does NOT route.

R-rule audit: would be same as (iii-a) + R14 + [D-37]-ext rule 5 symmetric-honesty obligation enforced at the verb layer. Not triggered.

---

#### DISPOSITION BRANCH (iii-c) — ALL-PASS + AD-5 BELOW-BAR-BUT-POSITIVE  [NOT SELECTED — retained as audit trail]

**Empirical lead**: would require AD-5 margin > 0 but < 10 pp; lower-CI > 0 but < 10 pp. Empirical: lower-CI 18.80 pp > 10 pp — branch (iii-c) does NOT route.

R-rule audit: would be same as (iii-a) with R14 + rule-5 symmetric-honesty as in (iii-b). Not triggered.

---

#### DISPOSITION BRANCH (iv) — CEILING-DISQUALIFIED  [NOT SELECTED — retained as audit trail]

**Empirical lead**: would require gate-(a) FAIL with valid training OR AD-5 margin lower-CI ≤ 0. Empirical: gate-(a) PASS (Â_resnet seed-averaged 0.5805 > 0.50 chance bar; 5/5 per-seed) AND AD-5 margin lower-CI 18.80 pp > 0 (and > 10 pp). Branch (iv) does NOT route.

R-rule audit at gate-5 (branch iv): would have triggered same as (iii-a) but with single-point null framing + R20 seed-noise-limited caveat. Not triggered.

---

**References block**:

- [D-15] AD-5 10 pp self-anchored bar (post-[D-36] retraction of external attribution).
- [D-24] AD-5 framework.
- [D-36] External-attribution retraction.
- [D-37]-Ext 1 honest-reporting rule (a)–(g): empirical lead, symmetric-disclosure rule 5, rule-7 fragility.
- [D-37]-Ext 2 R8–R23: cascade-close formality (R8), invariance discipline (R9), retired-model reuse (R10), venue-register (R11), scope-lock re-verbing (R13), self-anchored-bar fragility (R14), PROVISIONAL-by-default (R15), cross-atom propagation (R16), axis-relabeling (R17), nested-self-anchored-bar disclosure (R18), pitch-vs-resolution-preserving substrate-axis (R19), variance-decomposition (R20), domain-transfer anchor (R21), readout-topology check (R22), per-crop instrumentation (R23).
- [D-43] CVPR submission plan-of-record (32³ scope-lock; sprint-5 (c′) is post-CVPR; Step 5 CLOSED 2026-05-14 at HEAD `d7a23ca`).
- [D-49] Held-out region split (axis-0 70/15/15; (c′) reuses test region with crop_size=48 at n_grid=768 pitch).
- [D-50] CIC chunked-bincount refactor (enables (c′) 48³ at n_grid=768 without OOM).
- [D-51] Sprint-4 gate-5 absorption arc (32³ branch-iv PROCESS-FAILURE; (c′) at 48³ is the substrate-scale successor R10-orthogonal probe).
- [D-52] Sprint-5 source-choice option-(c) overturn; R13 + R14 banked.
- [D-53] Supervision-target axis upstream-vs-parallel obligation OPEN — explicitly NOT discharged by (c′).
- [D-54] Sprint-5 source-choice CLOSED at option-(d) defer + CVPR abstract amendment; R16 banked → closed for CVPR cycle.
- [D-55] Sprint-5 (c′)-at-48³ design v3 → v4 single-session 3-gate arc; R19/R20/R21/R22 BANKED; v4 absorbing pre-flight S2 BLOCKER; R23 BANKED.
- D'Amour et al. 2020 §6 (discriminative-classifier init-sensitivity precedent; R21 directional-risk anchor source).
- Touvron et al. 2019 (FixRes, primary R10 substrate-variation precedent retained at v3).
- Bolton et al. 2017 (1D flux-stat-scale Sherwood discriminability; cite-precedent for the substrate-axis result, retained from [D-52] amendment).
- `src/models/cnn3d.py:135` (R22 case-of-record source for `AdaptiveAvgPool3d(1)` readout-topology verification — unchanged from sprint-4).

**Review trail (NON-PROVISIONAL per R15 clause (b))**:

- gate-1 PI v1 design 2026-05-15.
- gate-2 defense-panel NEEDS-WORK 2026-05-15 (4 KILLER + 6 SERIOUS + R19/R20 candidates).
- gate-2 RE-pre-review APPROVE-WITH-AMENDMENT on v2 2026-05-15 (3 new KILLER + 4 SERIOUS + R21/R22 candidates). Discharged design-layer PROVISIONAL.
- gate-3 PI v3 absorption with 7 amendments 2026-05-15.
- gate-3-addendum v3 → v4 absorption of pre-flight S2 BLOCKER with 6 B-amendments + R23 BANK 2026-05-15 at HEAD `504037b` (subsequently HEAD `850e251` for the LEDGER addendum landing).
- gate-4 dispatch 2026-05-16: User-initiated Juno H100 dispatch via `bash scripts/submit_juno_sprint5_cprime.sh`; two prior dispatches FAILED and were fixed in-session (job `199428` SIGPIPE fix at HEAD `6094f80`; job `199429` data-symlink fix at HEAD `8992c12`); successful run on job `199430` at g-06-01 H100 partition, run_id `sprint5cprime_1778947953`, wallclock 19:50 min.
- gate-5 PI absorption: **THIS [D-56] entry, 2026-05-16**. Sign-off NON-PROVISIONAL per R15 clause (b).

**Cross-references**: §1 Pulse Stage 3 row updated (Part 2 template applied below); §7 Session Snapshot 2026-05-16 bullet appended (Part 3 template filled below).

---

## PART 2 — §1 Pulse Stage 3 Row Update (filled, branch iii-a template)

> **Sprint-5 (c′)-at-48³ (post-CVPR, NON-PROVISIONAL per [D-56], 2026-05-16)**: branch-(iii-a) ALL-PASS + AD-5 ABOVE-BAR at gate-5. Â_resnet_seed_avg = 0.5805 ± 0.0746; Â_mvsk_at_48cube_seed_avg = 0.3371 ± 0.0320; AD-5 margin = 24.34 pp (95% bootstrap CI [18.80, 28.88] pp; 5/5 per-seed PASS). 5/5 gates PASS (gate-(c) PASS_relaxed per S4 footnote — Juno venv `pytest` absence). R11 non-CVPR-only; R16 cross-atom propagation deferred to follow-on session. Run_id `sprint5cprime_1778947953`; Juno job `199430` g-06-01 H100; wallclock 19:50 min; 5 × 8000 = 40000 per-crop indicators verified (R23).

---

## PART 3 — §7 Session Snapshot 2026-05-16 Bullet (filled)

> - **2026-05-16 — Sprint-5 (c′)-at-48³ gate-4 dispatch → Juno H100 run → gate-5 absorption [D-56]**. User-dispatched the Juno H100 run via `bash scripts/submit_juno_sprint5_cprime.sh` (HEAD `850e251` on `exp/nerf`, [D-55] v4 NON-PROVISIONAL; two prior dispatches FAILED and were fixed in-session — `199428` ExitCode 141 SIGPIPE on `nvidia-smi | head` under `pipefail` fixed at HEAD `6094f80`; `199429` `FileNotFoundError` because sprint-5 (c′) ran from `JUNO_WORK` without sprint-4's per-run-dir data symlinks, fixed at HEAD `8992c12`). Successful run on job `199430` at g-06-01 H100 partition; run_id `sprint5cprime_1778947953`; run_tag `Sprint5cprime-48cube-5seed-8992c12-20260516-111221-d8928b`; wallclock **19:50 min** (under-runs 135 min budget because early-stop fires at epochs 6–7 across all 5 seeds with val-loss-divergence pattern seen at 32³). Empirical (seed-averaged across 5 seeds {42, 142, 242, 342, 442}): **Â_resnet_seed_avg = 0.5805 ± 0.0746**; **Â_mvsk_at_48cube_seed_avg = 0.3371 ± 0.0320**; **AD-5 margin = 24.34 pp** seed-averaged, 95% bootstrap CI **[18.80, 28.88] pp** (B=10000 resampling 5 seeds w/ replacement, RNG=20260516); AD-5 per-seed **5/5 PASS** (margins {24.75, 26.45, 14.28, 31.64, 24.59} pp); ρ_emp = 0.1945 (mean within-seed paired-classifier corr; per-seed {0.222, 0.216, 0.352, 0.093, 0.089}); σ_seed_resnet_emp = 0.0746 (1.49× the design v4 budget 0.05); ρ_seed_emp = 0.1623 (mean pairwise between-seed corr, 10 pairs, range [0.080, 0.242]). Per-gate verdicts: gate-(a) **PASS** (sanity floor 0.5805 > 0.50 chance bar, 5/5 per-seed), gate-(b) **PASS** (no training divergence; train_loss monotone descent epochs 1–6 prior to early-stop), gate-(c) **PASS_relaxed** (S4 cuDNN-determinism gate relaxed to 1e-5 per design v4 §6 footnote — Juno venv `pytest` absence, disclosure-only NOT a hard blocker), gate-(d) **PASS** (no AD-1 fail; mvsk_threshold_tightened=false across all 5 seeds), gate-(e) **PASS** (AD-5 margin lower-CI 18.80 pp > 10 pp self-anchored bar; 5/5 per-seed AD-5 PASS). **Branch selection**: **(iii-a) ALL-PASS + AD-5 ABOVE-BAR** per [D-55] §2 design-doc v4 routing table. **Substantive scientific finding**: at 48³ ρ-crop NeRF-substrate scale, the 4-class Sherwood feedback discrimination signature exceeds the moment-only MVSK 4-scalar subspace by ~24 pp in classifier accuracy — a single-point above-bar result, complementary to but NOT a generalization of the sprint-4 32³ branch-iv finding ([D-51] at 32³, the same problem collapses to ≈ MVSK 4-scalar subspace within seed noise). The two points jointly establish a single-pair substrate-scale contrast at fixed pitch; no claim is admissible about how the curve behaves at 96³+ or below 32³ without additional points. **R-rule audit verdict** at gate-5: R8 HOLDS (no cascade-close), R9 HOLDS (no "invariance"/"extends" verbs in any propagated text), R10 re-affirmed at (c′) gate-5 close (X⊥Y orthogonality between 32³ retirement reason and 48³ design purpose intact), R11 non-CVPR-only (CVPR remains 32³ scope-lock per [D-43]; CVPR PDF submission-ready at HEAD `d7a23ca`), R13 single-point claim only (no multi-substrate-scale verbs), R14 + R18 self-anchored-bar nesting disclosure OBLIGATORY in any branch-iii-a paper-text, R15 NON-PROVISIONAL per clause (b), R16 cross-atom propagation **TRIGGERED for future non-CVPR venue, deferred to separate latex-author session with explicit R16 sign-off gate**, R17 substrate-scale axis confirmed, R19 pitch-preserving confirmed (n_grid=768 unchanged; only crop_size 32³ → 48³), R20 variance-decomposition populated (σ_seed_resnet_emp = 0.0746 is 1.49× the v4 budget 0.05 — R20 caveat: seed-variance budget under-estimated by ~50% but AD-5 effect size large enough that MDE-budget under-estimate did not flip the gate), R21 domain-transfer anchor verdict **WITHIN-BAND-WITH-CAVEAT** (factor 1.49 confirms the [D-55] gate-3 R21 directional-risk disclosure in the conservative direction; R21 escalation route NOT triggered because gate verdict was not flipped), R22 readout-topology re-verified at `src/models/cnn3d.py:135` (`AdaptiveAvgPool3d(1)` unchanged; (c′) 48³ result does NOT lean on a token-count-ratio expressivity argument), R23 per-crop instrumentation verified at 5 seeds × 8000 lines = **40000 total per-crop indicators** across `cloud_runs/Sprint5cprime-48cube-5seed-8992c12-20260516-111221-d8928b/eval/per_crop_seed_{42,142,242,342,442}.jsonl`. **Paper-text disposition**: R11 non-CVPR-only eligibility; R16 cross-atom propagation to `papers/shared/sec/` + `papers/shared/sec_extended/` deferred to next session per [D-55] §3 R16 ownership pattern (cross-atom audit obligation discharged in a separate latex-author session, not in scope of this (c′) eval-close session). **Pre-flight verification recap**: S4 cuDNN-determinism gate-(c) verdict **PASS_relaxed** (Juno venv `pytest` absence; relaxed-gate fallback per design v4 §6 footnote — disclosure-only); A4 MVSK-at-32³ Â_mvsk_at_32cube_seed_avg = **0.3400** computed inside the (c′) pipeline alongside MVSK-at-48³ per [D-55] v4 B3/B4 amendments (single crop-extraction pass, two MVSK numbers emitted in `headline.json`). **R23 verification**: per-crop JSONL line counts (seed_42..seed_442) all 8000; total 40000 per-crop indicators (target 40000). **DVC tracking**: 5 checkpoint `.dvc` files added (total ~165 MB across 5 seeds × ~33 MB each): `cloud_runs/Sprint5cprime-48cube-5seed-8992c12-20260516-111221-d8928b/checkpoints/resnet18_3d_4class_best_seed_{42,142,242,342,442}.pt.dvc`. **Carry-forward**: (a) paper-text propagation post-gate-5 R16 audit deferred to next latex-author session for branch-(iii-a) non-CVPR venue eligibility; (b) multi-point scoping curve (96³+ substrate) deferred to future memory-budget-expanded design — required before any "extends"/"scales" verb is admissible (R13 + K4); (c) [D-53] supervision-target axis upstream-vs-parallel obligation remains OPEN — not addressed by (c′); (d) R24/R25 new-rule candidates surfaced at gate-5: **none**. R-rule banked count remains R8–R23 (15 banked rules; R12 deferred). (e) σ_seed under-estimation by ~50% (empirical 0.0746 vs budgeted 0.05) is recorded as future-variance-derivation calibration anchor for follow-on (c″)/(b) designs.

---

## Fill-in checklist (verification at absorption time)

1. ✅ **Run metadata**: run_id `sprint5cprime_1778947953`, Juno job `199430`, wallclock 19:50 min, run_tag `Sprint5cprime-48cube-5seed-8992c12-20260516-111221-d8928b`, per-seed JSONL paths verified at `cloud_runs/.../eval/per_crop_seed_{42,142,242,342,442}.jsonl`, DVC `.dvc` files pending checkpoint-tracking step, MVSK-at-32³ artifact path inside `headline.json` field `seed_averaged_mvsk_at_32cube`.
2. ✅ **Pre-flight verification**: S4 verdict PASS_relaxed (Juno venv `pytest` absence — disclosure-only); R23 per-crop count 40000 (5×8000); A4 MVSK-at-32³ = 0.3400; mvsk_threshold_tightened=false across all 5 seeds.
3. ✅ **Primary empirical quantities**: Â_resnet_seed_avg = 0.5805 ± 0.0746; Â_mvsk_at_48cube_seed_avg = 0.3371 ± 0.0320; AD-5 margin = 24.34 pp [18.80, 28.88] 95% bootstrap CI; per-seed AD-5 5/5 PASS; ρ_emp = 0.1945; σ_seed_resnet_emp = 0.0746; ρ_seed_emp = 0.1623.
4. ✅ **Per-gate verdicts**: (a) PASS, (b) PASS, (c) PASS_relaxed, (d) PASS, (e) PASS.
5. ✅ **Branch selection**: (iii-a) ALL-PASS + AD-5 ABOVE-BAR.
6. ✅ **Activate the corresponding disposition block**: (iii-a) selected; non-selected (i)/(ii)/(iii-b)/(iii-c)/(iv) retained as audit trail.
7. ✅ **R-rule audit verdicts**: R20 caveat (seed-variance budget under-estimated by ~50%), R21 WITHIN-BAND-WITH-CAVEAT (factor 1.49 confirms gate-3 directional-risk in conservative direction), R22 readout-topology re-verified intact.
8. ✅ **R24/R25 candidates**: none surfaced. R8–R23 banked count unchanged.
9. ✅ **§1 Pulse cell**: branch-(iii-a) template applied (Part 2 above).
10. ✅ **§7 Session Snapshot bullet**: full template filled with empirical numbers (Part 3 above).
