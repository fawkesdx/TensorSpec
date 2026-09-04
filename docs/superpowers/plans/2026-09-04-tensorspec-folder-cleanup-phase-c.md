# TensorSpec folder cleanup — Phase C (executed)

**Date:** 2026-09-04  
**Branch:** `TensorSpec_GUI`

## Scope actually done

| Item | Decision |
|---|---|
| Pure torch models (`maestroai_models.py`) | → `core/ml/models.py` |
| QThread workers (train/cluster/AL/align/loader) | **Stay in `gui/ml/`** — they import PySide6; `core/` stays Qt-free |
| Qt viewers / guides / tabs | Stay in `gui/ml/` |
| `core/peem_*.py` | → `core/peem/{bg,engine,roi,sumrule}.py` |
| Crystal suite panel split | **Skipped** (large UI rewrite, optional) |

## Why workers did not move

Design rule: `core/<domain>/` = engines/IO/math **without Qt**. Current ML “workers” subclass `QThread` and emit Qt `Signal`s. Moving them into `core/ml/` would force a Qt dependency into core. A future pass can split pure numerics out of the workers; that is not a rename.

## Import map

- `tensorspec.gui.ml.maestroai_models` → `tensorspec.core.ml.models`
- `tensorspec.core.peem_bg` → `tensorspec.core.peem.bg`
- `tensorspec.core.peem_engine` → `tensorspec.core.peem.engine`
- `tensorspec.core.peem_roi` → `tensorspec.core.peem.roi`
- `tensorspec.core.peem_sumrule` → `tensorspec.core.peem.sumrule`
