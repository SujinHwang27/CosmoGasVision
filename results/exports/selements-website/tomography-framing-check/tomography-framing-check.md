# PI framing-check — episode 01 "seeing-in-1d" §IV: is this "tomography"?

**Consumer:** selements-website (story-editor) · **Request:** framing-check on the word "tomography" + the CLAMATO/TARDIS precedent in primer ep01 (prose advice, no data owed) · **Responder:** CosmoGasVision PI · **Branch:** `service/data-export` (grounded on the `exp/nerf` LEDGER + spine state) · **Verb-ceiling:** [D-37] honest-reporting + the [D-73] close-out (binding).

> Boundary honored: this critiques selements' drafted §IV; it does **not** rewrite the primer. Scientific corrections are **binding**; storytelling preferences are **input**. The final wording is selements'.

**Verdict up front:** the primer's *instinct* is correct and honest — reserve the working-precedent verb for the z≈2–3 case and use a hedged "can a model reconstruct" for CGV. But the draft as written has **two hard citation-accuracy errors** and **one omitted confound** that make it fail [D-37] and the [D-73] verb-ceiling. All three are fixable with precision, not overhaul. The corrected passage is Group D.

---

## A. THE CATEGORY

**(1) The exact noun.** "Tomography" *is* the correct scientific category for CGV's **problem**, but not a safe label for CGV's **result** in a public primer.

- The problem class — recover a 3D field from many limited 1D line-of-sight projections — is legitimately called *Lyman-α forest tomography* (Lee+2018 and Horowitz+2019 both use that phrase in their titles). CGV's inverse problem is an instance of it.
- The method is a **neural field** (an MLP/voxel-grid trained through a differentiable Voigt/FGPA forward model) — in the literature, "machine-learning-based" or "neural implicit IGM reconstruction." The diagnosis-doc working title is exactly *"Trainability and optimization-failure characterization of neural implicit IGM reconstruction from Lyman-α sightlines at z = 0.3."*

So the exact noun for what CGV *attempts*: **a neural-field (learned) reconstruction of the 3D IGM density from Lyman-α sightlines** — an instance of the IGM-tomography problem class, using a method that is not classical tomography.

**(2) Should the primer say CGV's problem *is* tomography?** The problem is IGM tomography; that is accurate. But the honesty risk is decisive for a primer: CGV's headline finding is that this reconstruction is **under-constrained / fails to recover the 3D structure at z = 0.3** (the K2 degeneracy — the grid fits the observed flux ~4× *better* than the true field does, yet carries the wrong 3D structure). If a lay reader hears "CGV does tomography" right after "tomography demonstrably works (CLAMATO)," they import the precedent's *success* onto CGV — exactly the conflation [D-37] refuses.

**Endorsed as binding:** reserve the bare success-noun "tomography" for the *working* z≈2–3 precedent; use hedged, attempt-language ("can a model reconstruct…") for CGV. Do **not** call CGV "tomography" unqualified in this primer. If the category must be named for accuracy, name it *with the outcome attached* — e.g. "the same class of problem — IGM tomography — but at a redshift where we find it is under-constrained." Never the bare noun beside CLAMATO's success. The closing line ("Given only the sparse 1D flux, can a model reconstruct the 3D gas behind it?") is exactly right — keep it.

## B. THE PRECEDENT / ANCHOR

**(3) Is CLAMATO/TARDIS the right positive anchor?** Yes — as *context*, and only as context. The scope-lock uses CLAMATO/TARDIS as the "published work succeeds at z≈2–3" contrast and is emphatic that their r-values are **"context only, never our bar"** (R14/[D-36]). The primer may use them to establish the problem is solvable *somewhere*, motivating the z=0.3 question. It may **not** use them as a performance standard CGV is measured against. The draft does not cross that line — good.

**(4) Apples-to-apples? No — the omitted confound.** The draft's *only* stated difference is absorption depth ("much fainter absorption than those surveys had"). That is an **overclaim by omission**. Two axes differ:

- **Absorption depth** — real, correctly stated. At z=0.3, ⟨F⟩≈0.979 (~2% absorption); the low-z forest is information-sparse. CLAMATO operates where the forest is information-rich.
- **Sightline density / geometry** — the dropped confound. CLAMATO's success rests on a **dense** grid of backlights: mean transverse sightline separation **2.37 h⁻¹Mpc**, using faint star-forming galaxies (not just quasars) as backlights at 2.05<z<2.55 (verified first-hand, Lee+2018). That dense grid is *physically why* the field is constrained there. The sparse low-z QSO regime has no comparable density of faint-galaxy backlights.

**Internal caveat you must state honestly (cuts both ways):** CGV's *own* finding does **not** vary sightline density against CLAMATO. CGV holds sampling fixed at synthetic n_rays=1024 through a 60 cMpc/h box at z=0.3. So CGV isolates the **absorption-depth / low-z-information** axis at a *fixed synthetic geometry*; it does not test the density axis.
- The primer must **not** imply the only real-world difference from CLAMATO is faint absorption (density differs too — a physics error of omission).
- The primer must **also not** imply CGV *demonstrated* sparsity is the cause. Name density as the *observational* difference, while being clear CGV's synthetic test isolates the absorption axis and does not vary density. (Folded into Group D.)

## C. NAMING + CITATION

**(5) Keep the method generic ("a model") — correct for a primer.** "A model" is the right register for ep01. Naming "neural field / NeRF / MLP" is unnecessary for a lay reader and risks implying the *method choice* is the finding — it is not. The [D-73] close-out is a statement about the **problem** being under-constrained under this forward model (K2 is estimator-/architecture-independent), not about NeRF specifically. Endorsed.

**(6) "TARDIS… built for exactly this problem (Horowitz 2019)" — TWO citation errors, both binding.**

- **Error 1 — conflation of CLAMATO's method with TARDIS.** The draft implies the real-data CLAMATO map was built *using* TARDIS. It was not: CLAMATO's 2018 map was built with a **Wiener filter** ("a Wiener-filtered tomographic reconstruction … effective smoothing scale 2.5 h⁻¹Mpc," Lee+2018). TARDIS (Horowitz+2019) is a **separate, later, different method** — a constrained-realization / maximum-likelihood forward model, explicitly recovering "smaller scale structures than standard Wiener filtering." Not the method behind the 2018 map.
- **Error 2 — mock vs real data.** The draft says TARDIS is how "the cosmic web has been mapped … from real data." TARDIS Paper I **validated on MOCK data only** — "Applying this technique to mock Lyman-α forest data sets that simulate ongoing and future surveys such as CLAMATO, Subaru-PFS or the ELTs" (Horowitz+2019). Not applied to the real CLAMATO map in that paper.

Accurate statement: the *real-data* z≈2.3 cosmic-web map (CLAMATO, Lee+2018) was made with a **Wiener filter**; **TARDIS** (Horowitz+2019) is a *different, mock-validated* forward-model reconstruction method for the same problem class. (Pleasing internal consistency: CGV's *own* classical baseline is a Wiener filter — the same method class CLAMATO actually used. Name Wiener correctly and the primer's precedent and CGV's baseline line up truthfully.)

## D. CORRECTED SENTENCE(S)

Replace the §IV passage with the following. Same facts, right framing — the anchor now motivates the question without importing its success onto CGV, the CLAMATO/TARDIS methods are cited correctly, and both distinguishing axes (depth *and* density) are named honestly.

> It is different farther away. In the distant universe, around z ≈ 2–3, the gas is denser and the absorption far stronger, and the sightlines can be packed close together — surveys use faint background galaxies, not just quasars, as backlights, so the 1D pencils sit only a couple of megaparsecs apart. In that information-rich regime, reconstructing the 3D cosmic web from many sightlines demonstrably works: the CLAMATO survey (Lee et al. 2018) mapped it from real data using a Wiener filter, and later methods built for the same problem improve on that reconstruction — for example TARDIS (Horowitz et al. 2019), a forward-modeling approach so far demonstrated on simulated data. At z = 0.3 we face a much harder version of the same problem: the absorption is far fainter (only about 2% of the light is absorbed), and we do not have that dense grid of close-packed backlights. Our own experiment isolates the faint-absorption regime — synthetic sightlines through a simulated box at z = 0.3 — and asks whether the 3D gas can be recovered there at all.

Then keep the existing closing line unchanged:

> Given only the sparse 1D flux, can a model reconstruct the 3D gas behind it?

Two authoring notes (storytelling is yours; these precision points are binding):
1. "demonstrably works" attaches only to the **real-data CLAMATO/Wiener** clause — never to CGV. Keep it there.
2. If you foreshadow the finding, you may add after the closing question: *"The answer we find is: not from this flux alone — at z = 0.3 the 1D signal does not pin down the 3D field."* Consistent with K2 and within the verb-ceiling ("does not pin down," not "impossible"). Optional — but if you foreshadow, use that verb, not a stronger one.

---

## Literature verification log (confirmed vs not)

**Confirmed first-hand from primary sources:**
- **CLAMATO First Data Release (Lee et al. 2018), arXiv:1710.02894** — real-data map at **2.05<z<2.55**, built with a **Wiener filter** (2.5 h⁻¹Mpc smoothing); backlights are a mix of **faint star-forming galaxies and quasars** (240 sources); **mean transverse sightline separation 2.37 h⁻¹Mpc**.
- **TARDIS Paper I (Horowitz et al. 2019), arXiv:1903.09049** — a **constrained-realization / maximum-likelihood forward model**, distinct from and improving on Wiener filtering; **validated on MOCK data only** in this paper. Not the 2018 real-data map's method.

**Confirmed as a citation-accuracy error in the draft:** "mapped … from real data … using … TARDIS" conflates (a) CLAMATO's real 2018 **Wiener** map with (b) TARDIS's separate 2019 **mock**-validated forward model. Both the method-conflation and the mock-vs-real error are real; the corrected passage fixes both.

**Confirmed against CGV scope-lock (`D73_project_diagnosis_v2.md`):** z=0.3, ⟨F⟩≈0.979 (~2% absorption), P1, n_rays=1024 synthetic through a 60 cMpc/h box; CLAMATO/TARDIS context-only, never CGV's bar (R14/[D-36]); under-constraint at fixed synthetic sampling (K2 degeneracy) isolates the absorption/information axis and does not vary sightline density.

**Could not fully confirm (flagged, not load-bearing):** the exact *later* paper first applying TARDIS to *real* CLAMATO data — TARDIS Paper II (Horowitz+2020, arXiv:2007.15994) extends the framework, but not opened to verify first real-data application. Does not affect this ruling (the corrected passage says TARDIS Paper I was "so far demonstrated on simulated data," accurate to the 2019 paper). If the primer ever cites a *real-data* TARDIS result, cite the specific later paper and re-verify then.

---

## Sign-off

Good, honest primer instinct — the reserve-"tomography"-for-the-working-case choice is scientifically correct and **binding**. No overhaul needed. Three precision fixes are **mandatory**: (1) CLAMATO's real map used a **Wiener filter**, not TARDIS; (2) TARDIS Paper I is a **separate, mock-validated** method, not how real data was mapped; (3) name **sightline density** alongside faint absorption as a second real difference, while being clear CGV's synthetic test isolates the absorption axis and does not vary density. The corrected §IV passage (Group D) implements all three. The closing question stays as drafted.

Sources: CLAMATO Lee+2018 (arXiv:1710.02894); TARDIS Horowitz+2019 (arXiv:1903.09049); TARDIS II Horowitz+2020 (arXiv:2007.15994, flagged).
