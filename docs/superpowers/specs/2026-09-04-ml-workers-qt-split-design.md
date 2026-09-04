# Split Qt out of ML workers — design

**Date:** 2026-09-04  
**Branch:** `TensorSpec_GUI`  
**Choice:** Callback split (option 1)

## Problem

ML “workers” under `gui/ml/` subclass `QThread` and call `self.progress.emit` inside training/clustering/alignment loops. That blocks moving numerics into `core/ml/` (core must stay Qt-free).

## Goals

1. Pure compute in `tensorspec/core/ml/` with **no** PySide6 imports.
2. Thin `QThread` wrappers stay in `tensorspec/gui/ml/` and only translate callbacks → Signals.
3. Panel import paths keep working (`from tensorspec.gui.ml.… import TrainWorker` etc.).
4. Behavior unchanged: same signals, payloads, algorithms.

## Non-goals

- Rewriting model architectures (already in `core/ml/models.py`).
- Renaming `maestroai_*.py` files (can be a later cleanup).
- Replacing QThread with asyncio/process pools.
- Changing GUI progress UX.

## Pattern

```text
Panel  →  gui/ml/*Worker(QThread)
              run():
                core.ml.<job>.run_*(…, on_progress=lambda …: self.progress.emit(…))
                self.finished.emit(result)

core/ml/<job>.py
  def run_*(…, on_progress=None, on_model_changed=None):
      # pure torch / sklearn / numpy
      if on_progress: on_progress(value, message_or_loss)
      return results
```

`on_progress` defaults to no-op so core jobs are callable from tests/CLI without Qt.

## Module map

| Current (gui, thick) | Core (pure) | GUI wrapper (thin) |
|---|---|---|
| `maestroai_clustering.py` | `core/ml/clustering.py` | same path, `ClusterWorker` |
| `maestroai_training_ssl.py` | `core/ml/training_ssl.py` | same path, `TrainWorker` |
| `maestroai_training_sup.py` | `core/ml/training_sup.py` | same path, `SupTrainWorker` / `SupTestWorker` |
| `maestroai_active_learning.py` | `core/ml/active_learning.py` | same path, AL workers |
| `maestroai_alignment.py` | `core/ml/alignment.py` | same path, align workers |
| `maestro_loader.py` | *(keep thin — already delegates to `core.io`)* | optional: extract nothing beyond ensuring loader stays a shim |

Loader already calls `ARPESLoader` from core; leave as thin Qt wrapper unless a tiny `core` helper appears naturally.

## Progress callback contracts

Match existing Signal signatures so wrappers stay one-liners:

| Worker | Progress callback | Finished return |
|---|---|---|
| `ClusterWorker` | `(int, str)` | `(labels, umap_res)` |
| `TrainWorker` | `(int epoch, float loss)` + optional `on_model_changed(str)` | `dict` embeddings |
| `SupTrainWorker` | `(int, str)` | trained model |
| `SupTestWorker` | `(int, str)` | `prob_map` |
| AL workers | `(int, str)` | existing tuple maps |
| Align workers | `(int, str)` | `(map1, map2, map3, mode_str)` |
| `LoadWorker` | `(int, str)` | `(var_name, tensor_data)` |

Errors: core raises; GUI `run()` catches and `error.emit(str(e))` where that Signal already exists.

## Testing

1. Unit-test one core function without Qt (e.g. clustering on a tiny random array with a list-append `on_progress`).
2. Existing panel tests still import GUI workers and pass.
3. Smoke: import `MLSuite` + construct `ClusterWorker` / `TrainWorker`.

## Risks

- Long SSL/AL files: move bodies carefully; keep emit→callback mechanical.
- Accidental Qt import in `core/ml`: guard with a tiny test that fails if `PySide6` appears in `core/ml/*.py` AST/imports.
- Alignment/AL are large; do **clustering + supervised + loader** first in the same PR style commits, then SSL, then AL/align — still one overall feature, multiple commits.

## Success criteria

- `rg PySide6 tensorspec/core/ml` → no hits.
- All prior ML pytest files green.
- Panels unchanged except if an import path must point at a re-export (prefer zero panel edits).
