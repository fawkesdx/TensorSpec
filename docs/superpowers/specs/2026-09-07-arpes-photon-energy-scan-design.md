# ARPES photon-energy scan (kz / 4D Fermi) — design

**Date:** 2026-09-07  
**Repos:** TensorSpec_GUI (desktop ARPES suite) — not HTML  
**Status:** Approved (user: engine A; outer-loop A; label A; UI A; non-Chinook A; viewer A; Approach 1; §§1–3)  
**Related:** Chinook/Grizzly remote GPU path; `DataViewerPanel` N-D; Maestro motor-style axis naming

## Goal

Let ARPES Suite accept either a **single** photon energy or a **start / finish / step** array. Run Chinook (Grizzly) once per hv (outer loop), stack results so:

- **Dispersion geometry** (Θ or Φ degenerate / single entry): stacked cuts vs photon energy → simulate a **kz-style** measurement → 3D cube with `Photon Energy` axis.
- **Fermi / 3D geometry** (Θ and Φ both live): stacked Fermi maps vs photon energy → **4D** data.

`DataViewerPanel` already handles N-D; push stacked `TensorData` and browse there. Panel live plot stays single-hv-slice preview for Range runs.

## Non-goals (v1)

- SPR-KKR / three-step hv scan (UI disabled; remember for later)
- Engine-native batched ME across hv (Approach B deferred)
- N separate Einstein SSH jobs per hv (rejects GPU session reuse)
- Explicit kz conversion / Brillouin-zone display (hv axis only for now)
- HTML_web_app / HTML_einstein_app work
- Changing DataViewer architecture beyond consuming new axis labels

## Prerequisite (branch)

Before implementing this feature:

1. Point **`main`** at the current tip of **`TensorSpec_GUI`** (fast-forward; then push `origin/main` when authorized).
2. Leave **`HTML_web_app`** (and any HTML branch) untouched as a long-lived branch.
3. Continue feature work only on **`TensorSpec_GUI`**.

## Current state (as-is)

| Piece | Behavior |
|--------|----------|
| `photon_energy_spin` | Scalar `QDoubleSpinBox` only |
| Chinook kmesh | `hv` → `E_kin` → k-radius → K_bulk; ME shell tied to hv |
| Dispersion vs cube | Degenerate Θ/Φ in `arpes_panel` |
| Remote Einstein | One SSH job, one `--hv`, Grizzly CUDA + `--ngpus` |
| `DataViewerPanel` | Generic N-D (4D/5D OK); no hv cube axis from sim today |
| Sim push labels | `Energy`, `Θ (Slit)`, `Φ (Deflect)` |

## Approach (chosen)

**Approach 1 — GUI outer loop + remote one-job hv loop**

- Local: Python loop over existing single-hv Chinook path; stack.
- Einstein: **one** SSH job; runner loops hv inside; keep TB warm; rebuild ME shell per hv; same CUDA / multi-GPU path.
- Not Approach 2 (N submits) or Approach 3 (engine-native batch) for v1.

**Remember:** only **Chinook (Grizzly)** supports photon-energy-dependent scan now. **SPR-KKR later.**

---

## §1 — UI + hv array contract

**Where:** `tensorspec/gui/components/arpes_panel.py` photon energy row.

**Mode control:** combo `Single` | `Range` (default `Single` = current one spinbox).

**Range fields:** Start / Finish / Step (eV).

**hv list builder:**

```text
hv_list = np.arange(start, finish + step/2, step)   # finish inclusive on grid
```

Reject: `step <= 0`, `finish < start`, empty list. If `len(hv_list) > 64`, warn/confirm before run.

**Engine gate:** Range enabled **only** when backend is Chinook/Grizzly. Otherwise greyed with tooltip:

> hv-dependent scan = Chinook (Grizzly) only. SPR-KKR later.

**Payload:**

- Single: `experiment_kwargs["photon_energy"] = float` (unchanged).
- Range: `experiment_kwargs["photon_energies"] = list[float]`; keep `photon_energy` as first (or mid) value for backward-compatible metadata consumers.

---

## §2 — Run path (local + Einstein GPU)

**Geometry rule (unchanged):**

- Degenerate Θ or Φ → dispersion / kz-style stack.
- Both non-degenerate → Fermi 4D stack.

**Local Chinook (Range):**

1. Prefer load TB once across hv iterations when cheap.
2. For each hv: existing single-hv sim (rebuild k-mesh + ME shell; hv invalidates hoisted shell).
3. Stack along new axis 0: label `Photon Energy`, unit `eV`.
4. Output shapes:
   - Dispersion: `(Nhv, E, angle)` — surviving angle Θ or Φ.
   - Fermi: `(Nhv, E, Θ, Φ)`.
5. Progress text: `hv i/N (xx eV)`.

**Einstein remote (Range):**

- One SSH job (not N submits).
- Extend `chinook_remote_runner` with `--hv_start` / `--hv_finish` / `--hv_step` and/or `--hv_list`.
- Inside job: loop hv; keep TB; per hv rebuild ME + cube; use existing Grizzly CUDA, `select_gpu_ids`, `--ngpus`, radint prewarm.
- Write one stacked npz including `Photon Energy` axis.
- Fetch path reuses `on_simulation_finished` / workspace push.

**Single hv:** no behavior change.

**Important:** do **not** reuse one hoisted `me_shell` across different hv values (`exp.hv` must match shell).

---

## §3 — Viewer, IO, tests

**Viewer:** Push stacked `TensorData` to workspace / DataViewer. Panel live plot shows one hv slice (last or mid). Full hv browse in DataViewer (pick `Photon Energy` as slider dim).

**IO / workspace:** Extend `save_simulated_arpes` / simulated loader so npz supports 3D or 4D with `Photon Energy` axis; metadata includes `photon_energies` and engine=`chinook`.

**Tests:**

- hv list builder (inclusive finish; reject bad step).
- Stack shapes: dispersion → 3D; fermi → 4D.
- Range disabled when backend ≠ Chinook.
- Remote CLI parses hv range (arg parse / loop with mocked single-hv).

---

## Success criteria

- Single mode identical to today.
- Range + dispersion → 3D cube with `Photon Energy` axis; viewable in DataViewer.
- Range + Fermi → 4D; viewable in DataViewer.
- Einstein Range run uses **one** GPU job and existing multi-GPU machinery.
- Non-Chinook cannot enable Range; note visible for future SPR-KKR.
- `main` matches pre-feature `TensorSpec_GUI` tip; HTML branch untouched.

## Open follow-ups (not v1)

- SPR-KKR `ephot` scan
- Optional kz axis derived from inner potential
- Engine-native multi-hv ME amortization
