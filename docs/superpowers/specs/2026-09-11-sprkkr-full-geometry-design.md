# SPR-KKR full experimental geometry (manipulator + beam + analyzer) — design v0

Status: v3, 2026-09-12 -- Sandy supplied the real SPR-KKR manual (kept off Sandy's mac, never staged into this repo). Section 7 has manual-sourced facts + a correction + new leads.
Scope: SPR-KKR integration lane (see SPRKKR_GUI_THREAD.md two-lanes rule) --
this touches `tensorspec/core/dft/sprkkr/`, `tensorspec/core/arpes/one_step/kkr_wrapper.py`,
`tensorspec/gui/components/arpes_panel.py`. Nothing in OSKI/.

## 0. Where we are today

SPR-KKR (B3) currently receives, of the whole "Beam & Manipulator Geometry"
group:
- `work_function` -> `EWORK_EV` (used)
- `polarization` -> `POL_P` (used)
- `theta_ph` <- Beam Incidence (Lab) spinbox -- wired 2026-09-10, feeds `SPEC_PH THETA`
- the detector theta/phi *range* (kx/ky spinboxes) -> `SPEC_EL THETA={a,b}` / `PHI` (used)

It does **not** receive, and never has: Manipulator Theta/Azimuth/Tilt, Deflector
Angle, Slit Angle, Slit Width, `phi_ph` (`SPEC_PH PHI` is hardcoded 0.0 in
`ArpesParams`, `tensorspec/core/dft/sprkkr/params.py:90`). Those rows are now
hidden for B3 (`_sync_sprkkr_irrelevant_knobs`, `arpes_panel.py`) precisely
because they had zero effect.

Chinook (B1/GrizzlyME), by contrast, takes ALL of these and threads them
through a real lab-frame -> sample-frame -> bulk-frame rotation pipeline
before ever touching the tight-binding model:
- `compute_A_lab` / `photon_q_lab` build the light polarization vector and
  photon propagation direction in a fixed LAB frame from `incidence_angle`
  (measured from the surface normal, which is lab `+y` at zero manipulator
  angles -- see `photon_momentum.py` docstring).
- `build_k_bulk_mesh` turns each (slit-angle-rotated) detector (theta, phi)
  grid point into a vacuum photoelectron k-vector in that same lab frame.
- `sample_to_bulk_frame(vec, manip_theta, manip_azimuth, manip_tilt)` rotates
  any lab-frame vector into the crystal/sample frame via
  `R_total = R_z(manip_theta) @ R_y(manip_azimuth) @ R_x(manip_tilt) @ R_base`,
  `R_base = [[1,0,0],[0,0,1],[0,-1,0]]` (fixed axis relabel, not a manipulator
  angle) -- applied identically to `A_lab`, `K_LAB_VAC`, and (if
  `include_photon_momentum`) `q_lab`.
- `get_hkl_surface_frame(hkl, recip_matrix, azimuthal_ref)` gives the
  surface-normal-frame basis: `normal` (surface normal), `up_vector`
  (in-plane azimuthal reference -- defaults to the b* reciprocal axis
  projected into the surface plane, falling back to a* if b* is ~parallel to
  the normal). `X_surf x Y_surf x Z_surf` then maps sample-frame vectors into
  the bulk reciprocal frame the TB Hamiltonian actually uses.

So: the manipulator rotation is real physics in Chinook (it changes which
part of the lab-fixed beam+analyzer geometry lands on which part of the
crystal), and SPR-KKR is missing all of it. Setting the manipulator spinners
today changes nothing about an SPR-KKR run, which is exactly why they got
hidden rather than left dangling.

## 1. Why this isn't a drop-in field (the real gap)

SPR-KKR's ARPES task describes geometry entirely IN the crystal/surface
frame already: `MILLER_HKL` + `CRYS_VECS` fix the surface, `SPEC_PH
THETA/PHI` is the photon's incidence angle *relative to that surface*, and
`SPEC_EL THETA={a,b}` / `PHI` is the electron emission angle *relative to
that same surface* -- a polar-angle sweep at one fixed azimuth per job.
Chinook instead fixes the beam and analyzer in the LAB and rotates the
*sample* under them (manipulator), which is the actual experimental
picture (SPRKKR_GUI_THREAD.md already names the concrete case: "Exact
experiment geometry (slit at tilt -10.3 deg)" -- ARPES data taken at a real
manipulator tilt, currently approximated in the .inp by hand-setting
`SPEC_EL PHI=-10.3`, e.g. `scratch/sprkkr_gui_run/arpes_20260909_164942/arpes/arpes.inp`).

Converting one to the other means: take the lab-frame photon direction and
the lab-frame detector-grid direction, rotate BOTH into the sample frame
with the manipulator rotation Chinook already has, then re-express both
relative to the crystal's surface-normal frame (`get_hkl_surface_frame`) as
polar/azimuthal angles SPR-KKR understands. That conversion is buildable --
this doc scopes it -- but there is one structural mismatch no amount of
careful math removes:

**A straight lab-frame theta sweep (fixed deflector, slit sweeping) only
maps onto a straight SPR-KKR THETA-range-at-fixed-PHI sweep when the
manipulator rotation keeps the whole sweep in one crystal-frame azimuthal
plane.** For a pure tilt about the slit axis (the -10.3 deg case) that
approximately holds -- which is presumably *why* the ad hoc "just set PHI to
the tilt angle" hack has been giving usable results. For a general
manipulator setting (azimuth != 0, or tilt combined with a wide slit sweep),
the swept lab-frame line traces a *curved* path in (theta_crystal,
phi_crystal) space that a single `SPEC_EL THETA={a,b} / PHI=<one value>`
block cannot represent -- each theta point in the sweep would need its own
phi. SPRKKR_GUI_THREAD.md already flagged the real fix for this ("Next
options" #2): SPR-KKR's k-grid TASK keywords (`K1/K2/NK1/NK2`, referenced
there but never yet looked up in the SPR-KKR 9.7 manual by this codebase)
would let SPR-KKR sample an arbitrary k-parallel grid directly, matching
Grizzly's, instead of a polar (theta,phi) sweep at all. That is the correct
long-term answer; this doc treats it as a separate, larger phase 2 (needs
the SPR-KKR keyword semantics researched -- not scoped here) and proposes a
phase-1 approximation that is exact for the common case (pure tilt, or
small sweeps) and explicitly wrong (and detectable as wrong) outside it.

## 2. Proposed phase 1: single-reference-direction rotation

Compute two angle pairs per SPR-KKR job (not per detector point):

1. **Photon** (exact, always -- this part has no sweep to worry about,
   there's one beam): `q_lab = photon_q_lab(hv, incidence_angle)` -- CONFIRMED
   2026-09-11 from `ase2sprkkr/input_parameters/definitions/arpes.py`:
   `SPEC_PH THETA` = "Direction of the photon (the polar coordinate)",
   `PHI` = "...(the azimuth coordinate)" -- this is the propagation
   direction, i.e. `photon_q_lab`'s convention, NOT `compute_A_lab`'s
   polarization vector. Polarization is a fully separate keyword family
   (`POL_P` P/S/C+/C-, plus expert `ALQ/DELQ/ICIRC/IDREH`) and is already
   wired correctly via `pol_p` -- no change needed there.
   `q_sample = sample_to_bulk_frame(q_lab,
   manip_theta, manip_azimuth, manip_tilt)` -> project onto
   `get_hkl_surface_frame(hkl_abas, abas_cart)`'s (normal, up_vector,
   normal x up_vector) to get `theta_ph = angle from normal`, `phi_ph =
   azimuth from up_vector`.
2. **Electron / detector** (exact only at the sweep's *center* point, or
   when the sweep is effectively 1-D and stays in one plane -- see caveat
   above): take the detector direction at the midpoint of the configured
   theta range (deflector fixed per current SPR-KKR UI, since NP=1 is the
   common case per `_map_k_bounds`), rotate the same way, project onto the
   same surface frame, use the resulting `phi_e` as SPR-KKR's fixed
   `SPEC_EL PHI`, and use the *magnitude* of the rotation's effect on
   `theta_e` range endpoints (rotate both endpoints, not just the center) to
   set `SPEC_EL THETA={a,b}` in the crystal frame.

This reduces, for manip_azimuth=0 and a fixed-slit-axis tilt, to *exactly*
the existing "set PHI to the tilt angle" hack -- so item 1 of the validation
plan below is that reduction, checked numerically, not assumed.

Needs one new piece of code that does not exist yet: an in-plane reference
vector for the SPR-KKR side's `hkl_abas` (raw ABAS frame), analogous to
`get_hkl_surface_frame`'s `up_vector`, so `theta_ph`/`phi_e` land in the
*same* azimuthal convention already empirically confirmed for VTe2 ("Surface
basis 1 = 3.54 A axis (CIF b) -> SPEC_EL PHI=0 -> Gamma-K", SPRKKR_GUI_THREAD.md).
Concretely: `geometry.py`'s `_normal_hat(abas, hkl_abas)` already gives the
normal; it needs a sibling that gives the up_vector in the SAME (ABAS,
cartesian) frame, built the same way `get_hkl_surface_frame` builds its
`up_vector` (project b* into the surface plane, fall back to a*) so the two
codebases agree on what PHI=0 means for the same crystal. This should be
checked against the CIF-b-axis fact above before trusting it on a new
material.

## 3. What's still an open question (for Sandy, before code) -- ASKED 2026-09-11

1. Build phase 1 (single-reference approximation, immediately usable, wrong
   outside its stated regime) now, or go straight to researching the
   K1/K2/NK1/NK2 k-grid keywords (correct in general, unknown effort -- needs
   the SPR-KKR 9.7 manual, not in this repo)?
2. Is NP (deflector steps) ever > 1 in practice for SPR-KKR jobs today? If
   the detector sweep is always effectively 1-D (single deflector value),
   phase 1's "rotate both theta endpoints" approach is enough for the
   documented -10.3 deg case and probably most upcoming ones; if 2-D sweeps
   are coming soon, phase 1 is a short-lived stopgap and phase 2 should be
   prioritized instead.
3. Confirm `SPEC_PH THETA/PHI` really is the *photon propagation direction*
   (matches `photon_q_lab`'s convention) and not the light's polarization
   vector direction (matches `compute_A_lab`'s convention) -- these are
   different vectors in Chinook and only one of them is the right thing to
   rotate for `theta_ph`/`phi_ph`. This needs either the SPR-KKR manual or an
   empirical check (vary incidence_angle at manip=0, confirm the resulting
   ARPES intensity pattern moves the way a photon-direction change should,
   not a polarization change).
4. Validation gate before shipping either phase: reproduce the VTe2 -10.3 deg
   case and confirm phase 1's computed `SPEC_EL PHI` lands at (or very near)
   -10.3 with manip_tilt=-10.3, manip_azimuth=0, everything else at the
   values in `scratch/sprkkr_gui_run/arpes_20260909_164942/arpes/arpes.inp`.
   If it doesn't reduce to the known-good value, the rotation/frame
   convention is wrong and must not ship.

## 3a. Sandy's answers (2026-09-11)

1. **One cut at a time for now.** Fermi map (a deflection-range 2-D sweep)
   is a real future need, not now. -> Phase 1 (single-cut, single-reference
   rotation) is the right scope today. Fermi map later needs phase 2's
   k-grid keywords -- confirmed from `ase2sprkkr` source
   (`input_parameters/definitions/arpes.py`, `Section('SPEC_EL', ...)`) that
   SPR-KKR actually exposes **four** translating-vector pairs,
   `K1/NK1, K2/NK2, K3/NK3, K4/NK4` (plus a bare `KA`), not just two as
   SPRKKR_GUI_THREAD.md's older note assumed -- more room for a 2-D grid
   than expected, still needs the manual/empirical work to map Chinook's
   deflector grid onto them. Unchanged: phase 2 stays out of scope for this
   doc.
2. **Clarifies "azimuthal" = sample azimuth (manipulator azimuth), not
   detector NP.** Sandy's answer: the sample azimuth is used once, to orient
   the sample after the hkl surface is chosen -- not swept. The
   slit/detector geometry then stays fixed with respect to that oriented
   sample for the rest of the cut. Consequence for this design:
   `manip_theta/manip_azimuth/manip_tilt` are each a single static number
   per SPR-KKR job (not a function of detector angle), so `R_total` in
   section 2 is one fixed 3x3 matrix per job -- good, phase 1's
   "rotate the photon once, rotate both theta-sweep endpoints once" plan
   needs no per-point rotation. Important residual caveat, unchanged from
   section 1: a FIXED rotation applied to a whole swept line of lab-frame
   angles still traces a curved path in (theta_crystal, phi_crystal) unless
   the rotation is purely about the sweep-plane's own normal (the -10.3 deg
   tilt-only case). Sandy's "one cut at a time, sample azimuth set once"
   answer does not remove that caveat for a *general* (azimuth != 0 AND
   tilt != 0 combined) manipulator setting -- phase 1's endpoint-rotation
   approximation should flag (not silently swallow) any cut where the two
   endpoint phi values disagree by more than some small tolerance, since
   that disagreement is exactly the signal that a single PHI value is no
   longer a good approximation for that particular cut.
3. **Verified against source, 2026-09-11** (Sandy asked for this to be
   checked, not assumed): confirmed by folding the finding into section 2
   above -- `SPEC_PH THETA/PHI` is the photon propagation direction
   (`photon_q_lab`'s convention), not the polarization vector
   (`compute_A_lab`'s convention). Source: `ase2sprkkr`'s own input
   definition, `V('THETA', 45., info='Direction of the photon (the polar
   coordinate)')` / `V('PHI', 0., info='Direction of the photon (the
   azimuth coordinate)')`, `Section('SPEC_PH', ...)` in
   `ase2sprkkr/input_parameters/definitions/arpes.py`. Polarization is the
   separate `POL_P` (+ expert `ALQ/DELQ/ICIRC/IDREH`) keyword family,
   already correctly wired via `pol_p` -- no bug there, nothing to change.
4. **Confirmed: use the VTe2 -10.3 deg case as the phase-1 validation
   gate**, as stated in item 4 above.

## 4. Non-goals for this doc

- Not touching Fresnel/optical n,k, MFP, or "include photon momentum" --
  those have no SPR-KKR equivalent (one-step method has no separate
  transmission-coefficient or escape-depth-broadening step); staying hidden
  for B3 is correct, not a gap.
- Not attempting per-detector-point exact SPEC_EL fan-out in phase 1 (that's
  phase 2's k-grid approach, or would require one SPR-KKR job per angle
  point, which is a cost question of its own).


## 5. Phase 1 implementation attempt (2026-09-11) -- BLOCKED, needs Sandy input

Sandy said go. Before touching `geometry.py`/`kkr_wrapper.py`/`arpes_panel.py`,
ran the exact chinook rotation pipeline (`R_base`, manip `R_z@R_y@R_x`,
`R_hkl_to_bulk` from `up_vector=[0,1,0]` projected into the surface plane --
section 3a's confirmed convention) as a standalone numeric check against the
REAL VTe2 primitive lattice (`tests/sprkkr/fixtures/VTe2_prim.pot`,
`hkl_abas=(0,1,-1)`) and the documented -10.3 reference run
(`scratch/sprkkr_gui_run/arpes_20260910_062616/arpes/arpes.inp`:
`SPEC_PH THETA=45 PHI=0 EPHOT=84`, `SPEC_EL THETA={-15,15} PHI=-10.3`).
Script kept at `/tmp/validate_phase1.py` on Sandy's mac (not committed --
throwaway check, not production code).

**Finding 1 (fixed, minor): photon angle needs the supplement.** Rotating
the beam's propagation direction (`photon_q_lab`'s convention, confirmed
correct per section 3a Q3) through zero manipulator angles and projecting
onto the surface frame gives `theta=135 deg`, not the expected 45. But
`180 - 135 = 45` exactly. Makes physical sense: `SPEC_PH THETA` is the angle
between the surface normal and the direction the light ARRIVES FROM, i.e.
the supplement of the angle to the propagation direction (which points INTO
the surface, roughly antiparallel to the outward normal). Fix: use
`-beam_bulk` (or `180 - raw_theta`) for the photon side only. The detector
(electron) side does NOT need this flip -- boresight (lab theta=0,phi=0)
already comes out `theta_e = 0.0` at zero manip, i.e. normal emission,
correctly, with no supplement needed (photoelectron already travels roughly
along the outward normal). Confirmed by direct computation, not guessed.

**Finding 2 (real blocker): spherical (theta>=0, phi) is the wrong
extraction for what SPR-KKR's THETA range + one PHI actually means.**
SPR-KKR's `SPEC_EL THETA={a,b}` at fixed `PHI` describes a sweep through
normal emission along ONE fixed azimuthal half-plane, where NEGATIVE theta
is "the other side of the normal, same plane" -- not a separate azimuth.
A plain `arccos(dot(v,normal))`/`arctan2` read-off can't represent that: it
always returns `theta in [0,180]` and flips `phi` by 180 deg the instant the
sweep crosses the pole. Numeric check, tilt=0 (should reduce to the
already-correct existing behavior: `THETA={-15,15}`, one `PHI`, e.g. 0):
lab theta=-15 -> `(theta_e=15.0, phi_e=-90)`; lab theta=+15 ->
`(theta_e=15.0, phi_e=+90)` -- same magnitude, **opposite sign PHI**, not
one fixed PHI across the sweep. At tilt=-10.3 (the actual case this phase
was scoped to reproduce) it's worse, not better: lab theta=-15 ->
`(theta_e=18.13, phi_e=-123.7)`; lab theta=+15 ->
`(theta_e=18.13, phi_e=+123.7)` -- THETA range changed shape (not the
"unchanged range, just phi-shifted-by-tilt" picture the -10.3 hack assumed)
and PHI is nowhere near a single fixed value, let alone -10.3.

**Conclusion:** the curved-path caveat flagged in section 1 is not a small
effect outside a narrow regime -- it's first-order (90+ degree PHI swings)
even in the one case (-10.3, pure tilt, small +-15 sweep) this whole phase
was scoped around as the easy/validating case. A naive "rotate the two
sweep endpoints, read off spherical theta/phi" does not work. Need a real
SIGNED-theta / fixed-in-plane-azimuth extraction (find the sweep's plane
first, e.g. from the boresight-rotated normal, then get a signed angle
within that plane) before this is code, not a numeric-formatting detail.

**Two things needed from Sandy before continuing:**
1. For the actual -10.3 measurement: which of the three manipulator
   spinboxes (theta / azimuth / tilt) was the real nonzero one, and was the
   analyzer slit along lab x or lab z for that setup? (This check assumed
   `manip_tilt=-10.3`, slit along lab x, `manip_theta=manip_azimuth=0` --
   if the real DOF or slit axis differs, all these numbers are testing the
   wrong rotation and Finding 2 might look different, though Finding 1
   would likely still hold.)
2. Given how large this effect is even in the "easy" case: keep pushing
   phase 1 (build the correct signed-theta extraction, real but nontrivial
   extra code), or treat this as evidence phase 1 isn't worth it and go
   straight to researching phase 2's k-grid keywords (`K1-K4/NK1-NK4`)
   instead?


## 6. Sandy's deflector clarification (2026-09-11) + a bigger blocker

Sandy clarified the real hardware: "-10.3" was a **deflector** position (an
analyzer-side DOF, perpendicular to the slit), not a manipulator (sample)
rotation. She asked: is it true that (if slit=0/horizontal) deflector ~=
sample tilt, and (if slit=90) deflector ~= sample theta-rotation -- "and
does this hold in both SPR-KKR and Chinook."

**Chinook: confirmed directly from source, with a precision.** From
`build_k_bulk_mesh`: `K_SLIT = k*sin(THETA)`, `K_DEFL = k*sin(PHI)`,
`k_lab_x = K_SLIT*cos(slit) - K_DEFL*sin(slit)`,
`k_lab_z = K_SLIT*sin(slit) + K_DEFL*cos(slit)`. At `slit=0`: THETA moves
the lab-x component, deflector (PHI) moves the lab-z component only --
`manip_tilt` (`R_x`, chinook's tilt matrix) is exactly the rotation that
mixes lab-y/lab-z, so a small deflector move and a small tilt move the
detector direction the same way. At `slit=90`: THETA/deflector's roles on
lab-x/lab-z swap, and the equivalent manipulator axis becomes `manip_theta`
(`R_z`). So Sandy's physical picture matches Chinook's code exactly.
**Precision that matters for phase 1: this equivalence is only algebraically
exact right at the boresight (theta_lab=0), not across a finite sweep** --
checked directly: rotating `(K sin(theta_lab), sqrt(K^2-K^2 sin^2(theta_lab)), 0)`
about lab-x by angle `phi_lab` does NOT reproduce
`(K sin(theta_lab), sqrt(K^2-K^2 sin^2(theta_lab)-K^2 sin^2(phi_lab)), K sin(phi_lab))`
except at `theta_lab=0`. Doesn't matter in practice though: the exact,
correct thing to do is feed theta_lab AND the deflector angle directly into
Chinook's own `K_SLIT/K_DEFL/slit_angle` formula (that IS the literal
physical treatment of "deflector moves perpendicular to slit") -- no
manipulator-equivalence approximation needed for this step at all.

**SPR-KKR: cannot be confirmed from this codebase.** `ase2sprkkr`'s own
input definition (section 2 above) only documents `SPEC_EL THETA`/`PHI` as
"scattering angle" floats -- no formula tying them to `MILLER_HKL`/hkl axes
beyond that. No SPR-KKR manual or Fortran source is present in this repo or
on Sandy's mac (checked: no `*manual*`, `*.pdf`, `kkrspec` doc anywhere
findable). The only ground truth available is the empirical fact already in
SPRKKR_GUI_THREAD.md ("PHI=0 -> Gamma-K (CIF b axis), PHI=90 -> Gamma-M"),
which fixes what PHI means for TWO points on the azimuth circle but says
nothing about how an off-axis (deflector-shifted) cut's direction should
map onto SPR-KKR's own (THETA, PHI) grid.

**New, bigger problem found trying to answer this rigorously:** to actually
compute the SPR-KKR-frame answer, tried converting the *already-trusted,
already-in-production* baseline (bare theta sweep, deflector=0, i.e. the
`SPEC_EL THETA={-15,15} PHI=0.0` reference cut that's known correct) through
Chinook's own `K_LAB_VAC -> R_base -> R_hkl_to_bulk` pipeline and reading
off a spherical angle, exactly the same way section 5 read off the photon
angle. It does NOT reproduce `PHI=0`: it gives `phi_e = +-90` at the sweep
endpoints (real VTe2 lattice, `hkl_abas=(0,1,-1)`, matches the actual
b-axis-derived `up_vector`). Since this is the ALREADY-CORRECT case with
zero deflector involved, the mismatch means **Chinook's k-vector embedding
and SPR-KKR's native (THETA,PHI) parametrization are not the same
coordinate system** -- they agree on the high-level physics (theta=0 =
normal emission, phi measured from some in-plane azimuth reference) but not
on the specific 3-D embedding, and there is no way to derive SPR-KKR's exact
one from what is available in this repo.

**Where this leaves phase 1:** blocked on missing ground truth, not on
unsolved math. Three ways forward, all needing something this repo doesn't
have:
1. Get the actual SPR-KKR 9.7 manual (or the Fortran source that builds the
   ARPES angle grid, e.g. whatever computes `SPEC_EL`'s vectors from
   `THETA/PHI/MILLER_HKL`) and read the real formula off it.
2. Calibrate empirically: a handful of cheap SPR-KKR ARPES jobs at known
   small deflector-equivalent offsets, comparing the resulting cut's
   symmetry/band positions against expectations (the same way the
   PHI=0/90 -> Gamma-K/Gamma-M fact was itself established), to reverse-
   engineer the real embedding by data rather than by borrowed code.
3. Treat this as the sign to skip phase 1 entirely and go straight to
   phase 2's `K1-K4/NK1-NK4` k-grid keywords, if those turn out to
   sidestep the (THETA,PHI)-parametrization question altogether (an
   arbitrary k-parallel grid wouldn't need this conversion at all -- still
   unresearched, per section 0).

**Status: paused. Waiting on Sandy to pick 1/2/3 above, or supply the
manual/source if she has access to it.**


## 7. Real SPR-KKR manual read (2026-09-12) -- facts, one correction, two new leads

Sandy supplied the actual SPR-KKR/kkrspec manual (Oct 19, 2023 edition) from
her own machine. Per her instruction this file is NOT staged into the repo
or committed anywhere -- read directly off her Desktop via the device
bridge, extracted to this session's own scratch space only. Findings below
are cited by manual section, not re-derived by analogy this time.

**Confirmed: SPEC_PH THETA/PHI reference frame + sign (resolves Finding 1
properly).** Manual Sec. 6.2.2, structural-info figure: theta_e is defined
with respect to the surface normal n_hat; phi_e is defined with respect to
x; **theta_light is defined with respect to the z axis** (the kkrspec
structural stacking axis, increasing INTO the crystal from the surface) --
"if you define theta_light with respect to n_hat you should use -theta_n_hat
in the input file... normal incidence corresponds to theta_light=0". This
is the authoritative version of Finding 1's "supplement" guess: the exact
rule is theta_light_input = -theta_measured_from_outward_normal, not a bare
180-theta identity -- close in spirit to what Finding 1 found numerically,
but this is the real citation, not a numerical coincidence.

**phi_e's reference axis (x) is still not pinned down by the manual.** The
manual just says "with respect to x" and points to the 2D lattice vectors
a=(ax,ay), b=(bx,by) generated automatically for STRVER=1 -- i.e. `x` is
whatever in-plane axis ase2sprkkr's own struc.inp-generation step picks when
slicing the bulk potential by MILLER_HKL. That generation logic lives in
ase2sprkkr's Python source (not yet located/read) -- still the right next
step if phase 1's (THETA,PHI) route is pursued further, since Chinook's
`up_vector=[0,1,0]` convention (section 6) is now confirmed NOT to be it.

**CORRECTION to section 3a/6: retract the "SPEC_EL has 4 k-grid pairs"
claim.** `K1/K2/K3/K4` + `NK1..NK4` are real ase2sprkkr fields, but manual
Sec. 3.6 shows they belong to **TASK BLOCHSF** (the Bloch spectral function
A_B(E,k), a k-resolved DOS for band structure / Fermi-surface cuts) -- a
completely different calculation from ARPES (no photon, no matrix element,
no polarization). ARPES's own SPEC_EL section (manual Sec. 6.2.3) has no
K1-K4/NK1-NK4 at all: just THETA, PHI, NT, NP, POL_E, and (Expert Mode) TYP,
ISTR, POL0, POL0L, Q1-Q4. Phase 2 as "reuse K1-K4 for an arbitrary ARPES
k-grid" is likely not available -- ase2sprkkr's Python definition probably
just inherited/shares field names across task types; it does not mean the
ARPES Fortran routine consumes them. Whatever the real phase-2 answer is,
it isn't this.

**New lead 1 (manual-confirmed): SPEC_EL Expert Mode `TYP`.** Manual Sec.
6.2.3 + `ase2sprkkr/input_parameters/definitions/arpes.py`: `TYP` selects
the scan mode -- `0`=I(E) diagram, `1`=rotation diagram (phi scan), `2`=
scattering-angle diagram (theta scan, **the default, what every run so far
has used**), `3`=orthonormal projection, `4`=stereographic projection, with
the note "3,4 only for angular resolved... nt=np -> nx,ny". TYP 3/4 are a
**genuine native 2D angular map** (NT=NP reinterpreted as an (nx,ny) grid,
not a polar/azimuthal sweep) -- this is very likely the real mechanism for
the Fermi map Sandy wants eventually (section 3a Q1), and quite possibly
also a cleaner way to represent a single off-axis/deflector-tilted cut than
fighting the signed-theta/fixed-phi extraction in section 5/6, since a
projection grid sidesteps the polar-coordinate sign ambiguity entirely.
Nothing in this repo exercises TYP=3/4 yet -- unverified, needs a real test
run to see what it actually produces and how the projection axes are
defined.

**New lead 2 (ase2sprkkr-only, NOT in this manual -- unconfirmed, treat with
suspicion after the K1-K4 mistake above): `BETA1`/`BETA2`/`ROTAXIS`.** Found
in the same `arpes.py` Expert Mode block, right next to `TYP`: `BETA1`
("Begin of the rotation"), `BETA2` ("End of the rotation"), `ROTAXIS`
("Axis of the rotation", a 3-int array). If these do what their names say
-- rotate the electron-angle scan about an arbitrary crystal-frame axis over
a begin/end angle range -- this could be the exact, already-implemented,
native equivalent of "rotate the detector by the deflector angle about
whichever lab axis the slit orientation implies," making phase 1's whole
custom-rotation-math problem moot. But these three fields are **absent from
the manual entirely** (grepped, no match) -- either a newer kkrspec feature
this Oct-2023 manual predates, or (like K1-K4) a field that exists in the
Python wrapper without being functionally meaningful for TASK ARPES. Do NOT
trust this without an actual test run confirming it changes the .inp/output
sensibly.

**Status: paused, now on a different footing.** Before writing any more
design-doc math: worth a small, cheap empirical experiment -- run one
kkrspec ARPES job with `TYP=3` (or 4) and, separately, one with
`ROTAXIS`/`BETA1`/`BETA2` set, on the existing VTe2 pot, and just look at
what changes in the .inp/.out/.spc. That's a fast way to find out whether
either of these leads is real, before spending more effort on hand-deriving
angle conversions that keep turning out to test the wrong thing.

**Calibration runbook for `phi_offset_deg` (phase 1a):** see `docs/superpowers/runbooks/2026-09-13-sprkkr-phi-axis-calibration.md`.
