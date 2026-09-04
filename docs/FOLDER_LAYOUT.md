# TensorSpec folder layout

Rule of thumb for new code:

| Path | Put here |
|---|---|
| `core/<domain>/` | Engines, IO, math (no Qt) |
| `gui/suites/` | Thin suite shells only |
| `gui/components/` | Panel widgets; multi-tab domains may use `<domain>_tabs/` |
| `gui/ml/` | ML workers, guides, viewers, model stubs (Qt-adjacent) |
| `gui/services/` | Shared GUI helpers (auth, compute mode, cluster dropdowns, …) |
| `gui/main_browser.py` | App launcher / window registry |
| `plotting/` | Plot backends |

ML suite wiring today:

`gui/suites/ml_suite.py` → `gui/components/ml_tabs/*` → `gui/ml/session.py` → workers in `gui/ml/`.

## Parked (Phase C — needs a new approval)

- Move pure ML workers from `gui/ml/` → `core/ml/`
- Group PEEM engines: `core/peem_*.py` → `core/peem/`
- Optional: split fat Crystal suite into `components/crystal_tabs/`
