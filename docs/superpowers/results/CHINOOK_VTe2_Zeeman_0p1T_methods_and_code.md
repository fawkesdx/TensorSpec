# VTe2 (2̄01) one-step ARPES with Chinook + GrizzlyME in a 100 mT field — methods, parameters and code

Everything needed to reproduce the 40 cubes behind `Claude outputs/VTe2_B0p1T_vs_B0_grid.png`
(B = 0 vs B = 0.1 T, 2 beams × 2 slit geometries × 5 deflectors). Every number below comes
from the actual input files and logs of the production run, not from defaults or memory.

Generated 2026-09-26 from `scratch/vte2_zeeman/` (local) and
`<cluster>:<heavy_root>/chinook_gui_run/zeeman_0p1T/` (run host).
Question and plan: `docs/superpowers/specs/HANDOFF_magnetic_field_orbital_character.md`.

---

## 0. Result in one paragraph

A 100 mT in-plane field, fixed vertically down in the lab, changes the simulated ARPES
intensity by **1–9 × 10⁻⁸** per cut (Σ|I_B − I₀| / ΣI₀) and the per-deflector totals by
**~10⁻⁹**, in every beam/slit/deflector geometry. The change is real (a float64 recomputation
of one cut gives 7.7 × 10⁻⁸ vs 7.5 × 10⁻⁸ from the float32 production cube; float32 storage
alone gives 2 × 10⁻⁸) and it is structured — it follows the bands — so the field does alter
orbital character and matrix elements. But it is ~10⁴–10⁵ below what an ARPES measurement
resolves. B along the long in-plane (CDW) axis acts 2–4× more strongly than B along b.
From the B² scaling of band mixing (§8), a ~1 % change in the geometry dependence would
need a field of order 30–100 T in this model. **Within this model the hypothesis is
bounded, not supported, at 100 mT.**

---

## 1. Software

| Component | Version / identifier |
|---|---|
| Chinook | as installed in the cluster's TensorSpec Python env |
| GrizzlyME | `fawkesdx/GrizzlyME` (GPU matrix elements, diagonalisation, spectral) |
| PyTorch | 2.7.1+cu118 |
| GPU | 1 × Tesla V100-SXM2-32GB (`--ngpus 1`; host shared) |
| Runner | `tensorspec/core/arpes/one_step/chinook_remote_runner_template.py` at commit `b52e05e` |
| k-mesh / ME driver | `tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py` at commit `046b7e9` |
| Zeeman injection | `scripts/chinook/add_zeeman.py` (this study; §3) |

The two commits are fixes made for this study and are required to reproduce it:

- `b52e05e` — energy window padded by 10 × se_width (0.505 eV) and cropped before saving.
  Without it, eigenstates just outside the window are dropped while their Lorentzian tails
  belong inside, and the cube jumps 10–170× between neighbouring θ.
- `046b7e9` — `physics["spinor_basis"] = "interleaved"` puts the up and down spinor components
  in separate matrix-element rows, so the two photoelectron spin channels add in intensity
  rather than amplitude. Negligible at B = 0 (Kramers cancellation, ~3 × 10⁻⁴), but required
  once a field splits the pairs.

---

## 2. Structure and tight-binding model

VTe2 1T″ CDW, C2/m, conventional cell a = 14.175 Å, b = 3.541 Å, c = 9.058 Å, β = 110.55°
(QE-relaxed, PBE+D3). The QE and XRD cells are the same setting, so (−2 0 1) is the same
plane in both.

| Item | Value |
|---|---|
| Source | QE 7.5 PBE + SOC (`noncolin`, `lspinorb`), Wannier90 `spinors = true` |
| Model | `tb_data.npz`, md5 `e24c2cbfe6170c9d0cf73d26e1bd4e1b` (identical local and on the cluster) |
| Basis | 324 spinor WFs = 162 orbitals × (up, down), **interleaved** (index 2m = orbital m up, 2m+1 = down; checked: labels pair) |
| Orbitals | V 4s 4p 3d, Te 5s 5p + d shell; listed atom by atom (s, p z/x/y, d ZR/xz/yz/XY/xy) |
| Hoppings | 1 671 856, \|t\| > 10⁻⁴ eV, E_F (12.4202 eV) folded into on-site terms |
| Spin frame | QE Cartesian z (default for Wannier spinors); npz `a_mat` equals QE `CELL_PARAMETERS` |

---

## 3. The field term

Added on-site (R = 0) to every orbital, identical for both spin components where it is orbital:

    H_Z = μ_B B · (L + 2S) = μ_B B n̂·σ  +  μ_B B n̂·L          μ_B = 5.7883818 × 10⁻⁵ eV/T

- **Spin part**: 2 × 2 block μ_B B [[n_z, n_x − i n_y], [n_x + i n_y, −n_z]] on each (2m, 2m+1).
  At 0.1 T the bare splitting 2μ_B B = 11.6 μeV.
- **Orbital part**: L on each atom's p and d shells, built by applying L = −i r × ∇ to the
  real-harmonic polynomials named by the Wannier labels (x, y, z; xy, yz, xz, (x²−y²)/2,
  (3z²−r²)/(2√3)), so no Y_lm phase convention enters. Verified: Hermitian,
  [L_x, L_y] = i L_z to 2 × 10⁻¹⁶, L² = l(l+1), eigenvalues −l…l. 36 p/d shells.
- Rows are **appended** to the npz; the builders accumulate duplicate (i, j, R) entries.
  The original rows are untouched; B = 0 reproduces the original Hamiltonian and cube
  bit-for-bit (max diff 0.0).

Files (0.1 T, spin + orbital):

| File | Field in crystal Cartesian | Crystal axis | md5 |
|---|---|---|---|
| `tb_B0p1_az90.npz` | (0.7107, 0, 0.7035) | long in-plane (CDW) axis | `fa450066ecef275655af2c80d648b544` |
| `tb_B0p1_az0.npz`  | (0, 1, 0) | b (3.54 Å) | `6b2726bd5dee3bc7c0f17849f0c5d8f3` |

```bash
python scripts/chinook/add_zeeman.py --tb tb_data.npz \
    --physics runs/phys_th55_ph0.json  --b-tesla 0.1 --dir lab:0,0,-1 --out tb_B0p1_az90.npz
python scripts/chinook/add_zeeman.py --tb tb_data.npz \
    --physics runs/phys_th55_ph90.json --b-tesla 0.1 --dir lab:0,0,-1 --out tb_B0p1_az0.npz
```

Hamiltonian-level checks (k = (0.13, 0.41, 0.27), states in −1…0.1 eV):

| B | pair splitting (median) | weight mixed into other bands (max) |
|---|---|---|
| 0 | 25 μeV (Wannier-model symmetry noise) | — |
| 0.1 T | 26–31 μeV | 6.5 × 10⁻¹⁰ – 1.0 × 10⁻⁹ |
| 1 T | 111–126 μeV | 6.5 × 10⁻⁸ – 1.0 × 10⁻⁷ |
| 10 T | 1.1 meV | 6.5 × 10⁻⁶ – 1.0 × 10⁻⁵ |
| 100 T | 11 meV | 6.7 × 10⁻⁴ – 1.0 × 10⁻³ |

Mixing scales as B², as perturbation theory requires. Band positions move by < 7 meV even
at 20 meV splitting, far below the 50 meV linewidth — consistent with the measured
unchanged dispersion.

---

## 4. Geometry

Cleave (−2 0 1), `"hkl": [-2, 0, 1]` in every physics JSON. Surface normal
−2a* + c* = (−0.7035, 0, 0.7107) in crystal Cartesian.

Chinook lab frame: surface normal +y, plane of incidence x–y, slit 0 along x (horizontal),
lab z vertical. TensorSpec convention for the two slit geometries (from
`scripts/chinook/make_physics_json.py`):

| Tag | Beam from normal | manip_azimuth | slit_angle | Slit lies on | B (lab −z) lies on |
|---|---|---|---|---|---|
| th55_ph0 | 55° | 90 | 0 | b | long axis — **B ⟂ slit** |
| th20_ph0 | 20° | 90 | 0 | b | long axis — **B ⟂ slit** |
| th55_ph90 | 55° | 0 | 90 | b | b — **B ∥ slit** |
| th20_ph90 | 20° | 0 | 90 | b | b — **B ∥ slit** |

The field is fixed in the lab (vertical down) and rotated into the crystal with exactly the
transform TensorSpec applies to the light vector (`sample_to_bulk_frame` + the hkl surface
frame), so it follows the sample azimuth. It lies in the cleave plane in both geometries
(90.0° to the normal). Deflector = the collapsed k_bounds['Y'] range, −10.3, −5, 0, +5, +10.3°.

---

## 5. Run matrix

| Parameter | Value |
|---|---|
| B | 0 and 0.1 T |
| hv, work function, V₀, T | 84 eV, 4.5 eV, 12 eV, 10 K |
| Polarisation | Linear Horizontal (p), incidence 55° or 20° |
| Slit angle | −10…+10°, 100 points |
| Energy | −1.5…+1.0 eV, 501 points (5 meV step); padded internally ±0.505 eV (101 steps) |
| Broadening | se_width 0.05 eV, res_E 0.005 eV, res_k 0.005 Å⁻¹ |
| Matrix elements | Full, rad_type slater, mfp 8.7 Å, spinor_basis interleaved |
| Engine | GrizzlyME, `--device cuda --layout full --theta_chunk 50` |
| Not included | Fresnel, photon momentum, k_z broadening, field acting on the outgoing photoelectrons |

### Commands

Bundle `scratch/vte2_zeeman/remote_bundle/` (runner, kmesh, schedule, fresnel,
photon_momentum, 4 physics JSONs, 2 field models, `run_zeeman.sh`) copied to
`chinook_gui_run/zeeman_0p1T/`; B = 0 uses `../tb_data.npz`.

```bash
ssh <user>@<cluster> "cd <heavy_root>/chinook_gui_run/zeeman_0p1T && nohup ./run_zeeman.sh > run_zeeman.log 2>&1 < /dev/null &"
```

One cut, as issued by `run_zeeman.sh`:

```bash
python -u chinook_remote_runner.py --tb_file tb_B0p1_az90.npz --physics_file phys_th55_ph0.json \
  --theta_min -10 --theta_max 10 --ntheta 100 --phi_min -10.3 --phi_max -10.3 --nphi 1 \
  --e_min -1.5 --e_max 1.0 --ne 501 --hv 84 --workf 4.5 --v0 12 --temp 10 \
  --engine grizzly --device cuda --layout full --ngpus 1 --theta_chunk 50 \
  --out_file cuts/B0p1_th55_ph0_defl_m10p3.npz
```

### Cost

GPU compute 9.6 s per cut (mean of 40); ~20 s per cut including model load; 40 cuts ≈ 14 min.

---

## 6. Post-processing

| Output | Location |
|---|---|
| Raw cubes (θ, φ, E) | `scratch/vte2_zeeman/cuts_remote/{B0,B0p1}_{cfg}_defl_{tag}.npz` |
| Viewer npz, per cut + stacked (kx, ky, E) | `scratch/vte2_zeeman/viewer/{B0,B0p1}/cuts_vte2_{cfg}_chinook/` |
| MATLAB (value[100×5×501], x slit °, y E−E_F, z deflector °) | `scratch/vte2_zeeman/matlab/vte2_cuts_{cfg}_chinook_{B0,B0p1}.mat` |

Viewer and .mat metadata carry `zeeman_b_tesla`, `zeeman_nhat_crystal_cart`,
`zeeman_axis_in_crystal`, `spinor_basis`, and `far_field_B_on_photoelectrons: not included`.
Converters: `scripts/chinook/to_viewer_npz_chinook.py` (`convert`) +
`tensorspec/core/dft/sprkkr/viewer_export.stack_viewer_cubes`; `scripts/sprkkr/to_matlab.py`.

---

## 7. Figure

`Claude outputs/VTe2_B0p1T_vs_B0_grid.png`: 8 rows × 5 deflectors. For each geometry, the B = 0
spectrum (√ scale) and below it (I(0.1 T) − I(0)) scaled to its own maximum; the printed
number is the true maximum |ΔI| / I_max. Under B ⟂ slit the differences follow the bands in
paired red/blue lobes (small weight transfer); under B ∥ slit they are weaker and in the
brightest regions approach float32 storage noise.

---

## 8. Numbers

Σ|I_B − I₀| / ΣI₀ per cut (B = 0.1 T):

| Geometry | −10.3° | −5° | 0° | +5° | +10.3° |
|---|---|---|---|---|---|
| beam 55°, az 90 / slit 0 (B ⟂ slit) | 5.9e−8 | 6.8e−8 | 7.5e−8 | 5.2e−8 | 4.8e−8 |
| beam 20°, az 90 / slit 0 (B ⟂ slit) | 6.6e−8 | 8.4e−8 | 8.9e−8 | 6.0e−8 | 3.8e−8 |
| beam 55°, az 0 / slit 90 (B ∥ slit) | 2.0e−8 | 3.5e−8 | 3.5e−8 | 2.8e−8 | 1.7e−8 |
| beam 20°, az 0 / slit 90 (B ∥ slit) | 1.3e−8 | 2.2e−8 | 1.3e−8 | 1.9e−8 | 1.4e−8 |

Per-deflector integrated intensity at B = 0 (the relative change with B is ≤ 1 × 10⁻⁸ everywhere):

| Geometry | −10.3° | −5° | 0° | +5° | +10.3° |
|---|---|---|---|---|---|
| beam 55°, az 90 / slit 0 | 1.358 | 1.061 | 0.719 | 1.043 | 2.206 |
| beam 20°, az 90 / slit 0 | 0.330 | 0.267 | 0.256 | 0.443 | 0.715 |
| beam 55°, az 0 / slit 90 | 1.729 | 1.064 | 0.581 | 1.751 | 4.505 |
| beam 20°, az 0 / slit 90 | 0.863 | 0.943 | 0.869 | 1.588 | 3.342 |

Ratio (az 90 / slit 0) / (az 0 / slit 90): beam 55° 0.785 0.998 1.238 0.596 0.490, beam 20°
0.382 0.283 0.295 0.279 0.214; change with B ≤ 1 × 10⁻⁸.

Precision check (config th55_ph0, deflector 0, recomputed locally in float64, not saved as
float32): Σ|ΔI|/ΣI = 7.69 × 10⁻⁸, total change +8.9 × 10⁻⁹; float32 rounding alone 2.2 × 10⁻⁸.

---

## 9. Caveats to state in the paper

1. **Not self-consistent.** The Zeeman term is imposed on a fixed tight-binding model. Exchange
   enhancement of the induced moment (Stoner) is absent. The SPR-KKR SCF with M allowed
   converged to M = 0, so no large enhancement is expected, but the self-consistent `BEXT`
   calculation (handoff §6) is the test if a field effect is ever observed.
2. **Orbital term from labels.** L assumes each Wannier function is the atomic orbital its
   label names. On some V sites the p/d Wannier centres sit up to 0.5 Å off the atom
   (hybridised functions), so μ_B B·L is approximate there.
3. **Model noise vs field.** The Wannier model's own Kramers splitting (~25 μeV) exceeds the
   0.1 T spin splitting (11.6 μeV); the within-pair reorientation is therefore not resolved,
   but it is invisible in ARPES anyway (the pair-summed intensity is basis-independent).
4. **k_z sensitivity.** Chinook's free-electron final state at one k_z: the ±deflector ratio
   moves from 6.2 to 0.6 as V₀ goes 12 → 32 eV. The B-on/B-off comparison is at fixed geometry
   and V₀, so it is unaffected, but absolute geometry ratios are not robust.
5. **No field on the photoelectrons.** The 100 mT field bends 80 eV electrons with a radius of
   ~0.3 mm on the way to the analyser; that instrumental effect is not modelled.
6. **SPR-KKR not used here.** Its ASA potential misplaces bands (graphite test in
   `scratch/sprkkr_graphite/`, and the WTe2 FULLPOT comparison in `scratch/wte2/results/`), and
   the VTe2 runs used the auto-picked `IQ_AT_SURF = 2`, which the WTe2 work found builds the
   crystal the wrong way up. SPR-KKR full-potential remains the route for a self-consistent
   moment table.
