# DFT Fat Bands — Design Spec

Date: 2026-08-07  
Status: approved for implementation  
Related plan: DFT Fat Bands Implementation Plan

## Problem

DFT Suite HTML has a disabled **Fat Band Target** control. Chinook already returns eigenvectors `(nk, norb, nband)` and orbital labels, but the browser never gets character weights. Lab teaching needs orbital-projected bands (e.g. graphene π = pz) without re-solving H(k) on every target change.

## Goals

- Enable fat-band selection: **group presets** (s/p/d, by element) **and** individual orbitals.
- Changing target **re-projects** cached eigenvectors only (no re-diagonalize).
- Do **not** ship full eigenvectors to the browser.
- Plot intensity like unfold weights (amber scatter / alpha).

## Non-goals

- Client-side eigenvector download
- Fat projection on 2D isoenergy meshes
- Restoring Qt GUI fat-band UI

## Design

### Cache

`POST /api/dft/{name}/bands` already calls `workspace.push_band_structure` with eigenvectors. Extend the payload with `orbital_labels` so fat projection can match indices after the solve.

### Target encoding

| Value | Meaning |
|-------|---------|
| `none` | Clear fat weights |
| `shell:s` / `shell:p` / `shell:d` | Match orbital character (SOC `_up`/`_dn` ok) |
| `element:C` | Labels starting with `C_` |
| `orbital:C_pz` or bare `C_pz` | Exact label |

### Math

```
probs = |evecs|²
fat_weights[k,n] = Σ_{i ∈ idxs} probs[k,i,n]
```

Clip to `[0, 1]`. API shape matches unfold `weights`: per-band lists of k-values.

### API

`POST /api/dft/{name}/bands/fat` with `{ "fat_target": "shell:p" }`  
Resolves bands node `{name}_bands` (or `name` if already a bands node). Returns `fat_weights` + target metadata.

### UI

After calculate: populate `#tb-fat` with `<optgroup>` Shells / Elements / Orbitals. On change: call fat endpoint, merge `fat_weights` into last plot result, redraw. No full recalculate.

### Plot

- `fat_weights` only → guide lines + amber orbital-weight scatter  
- unfold `weights` only → existing unfold scatter  
- both → unfold points with alpha × fat  
- none → plain lines  

## Success criteria

- Unit tests for index resolve + weight sum  
- Smoke: bands → fat `shell:p` without second `solve_bands`  
- Dropdown enabled after calculate; switching target updates plot  
