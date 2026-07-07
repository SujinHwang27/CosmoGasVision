# selements-website export registry (CosmoGasVision)

Every artifact shipped to the selements-website consumer is registered here.
Owner: **data-engineer** (primary). One row per serviced request. This export
workflow is cross-cutting infrastructure — it lives on the dedicated
`service/data-export` branch and is **not** part of any experiment track or its
LEDGER decision-numbering. See `.claude/skills/data-export/SKILL.md` for the full
contract and the "service a new request" recipe.

**selements-website** curates project sources across the author's projects;
CosmoGasVision is one. Any **claim-bearing** export MUST respect the [D-73]
close-out verb-ceiling (characterization at z=0.3; the flux inverse problem is
"under-constrained under this FGPA forward model"; the production MLP "fails 2 of 3
directly-evaluated [D-13] gates" with the 3D ξ gate "never evaluated on the MLP";
the ξ-0.6-gate is demoted; no claim beyond z=0.3 — CLAMATO/TARDIS succeed at
z≈2–3). Lead with the honest observation; never a reconstruction / "it works"
claim. Sources: `experiments/nerf/LEDGER.md` §3 [D-37] + [D-73] amendment-9/10;
`experiments/nerf/design/D73_project_diagnosis_v2.md`.

| request-slug | producing function | source-data path | git SHA at export | date | consumer-facing filename | caveats |
|---|---|---|---|---|---|---|
| narrative-arc-honesty-review | project-architect (PI) narrative-arc honesty review (prose; no code export) | `experiments/nerf/LEDGER.md` §3 [D-73] am-9/am-10 + `experiments/nerf/design/D73_project_diagnosis_v2.md` §3a/§4/§5/§6 | 1b80f56 | 2026-06-29 | `narrative-arc-honesty-review.md` | **Claim-bearing — PI sign-off = APPROVE-WITH-BINDING-CORRECTIONS on selements' drafted 4-episode arc.** Prose advice only (no data/figure owed); critique not rewrite — scientific corrections binding, storytelling input. Binding caveats the narrative MUST carry: K2 4× margin includes integrator-induced slack (0.0101 = OUR FGPA-vs-RT error, not physical); "~25%" is ceiling-relative under a DEMOTED ξ estimator (vs 0.0298, never vs 0.6); grid-vs-MLP P_F magnitudes (0.0352 vs 0.4155) NOT a same-config contrast (n_rays 1024 vs 64) though verdicts are; under-constraint scoped to z=0.3 under this forward model — NOT "the limit is the information," NOT a claim about nature, NOT "tomography impossible"; MLP fails 2 of 3 gates (3D ξ never evaluated on MLP); single realization, Sherwood P1, 60 cMpc/h box. Full sidecar: `narrative-arc-honesty-review.provenance.json`. |
| tomography-framing-check | project-architect (PI) framing-check ruling (prose; no code export; literature verified first-hand via WebFetch) | `experiments/nerf/design/D73_project_diagnosis_v2.md` §1/§3c/§4/§5/§6 + CLAMATO Lee+2018 (arXiv:1710.02894) + TARDIS Horowitz+2019 (arXiv:1903.09049) | 9e93b85 | 2026-07-07 | `tomography-framing-check.md` | **Claim-bearing + public-facing — PI ruling on the word "tomography" + CLAMATO/TARDIS in primer ep01 §IV.** Prose advice only; critique not rewrite. Verdict: reserve "tomography" for the working z≈2–3 case (BINDING), never for CGV's result (under-constrained at z=0.3). Three MANDATORY citation/physics fixes: (1) CLAMATO's real-data map used a **Wiener filter**, NOT TARDIS; (2) TARDIS Paper I is a **separate, MOCK-validated** method, not how real data was mapped; (3) name **sightline density** (CLAMATO 2.37 h⁻¹Mpc, faint-galaxy backlights) alongside faint absorption as a second real difference, while stating CGV's synthetic n_rays=1024 test isolates the absorption axis and does NOT vary density. Corrected §IV passage supplied; closing line kept. CLAMATO/TARDIS are context-only, never CGV's bar (R14/[D-36]). Full sidecar: `tomography-framing-check.provenance.json`. |
