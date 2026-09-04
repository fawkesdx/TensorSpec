# TensorSpec folder layout

Rule of thumb for new code:

| Path | Put here |
|---|---|
| `core/<domain>/` | Engines, IO, math (no Qt) |
| `gui/suites/` | Thin suite shells only |
| `gui/components/` | Panel widgets; multi-tab domains may use `<domain>_tabs/` |
| `gui/ml/` | ML Qt workers (`QThread`), guides, viewers, model-warehouse tabs |
| `gui/services/` | Shared GUI helpers (auth, compute mode, cluster dropdowns, …) |
| `gui/main_browser.py` | App launcher / window registry |
| `plotting/` | Plot backends |

## Domain packages today

- **ML:** `core/ml/{models,clustering,training_ssl,training_sup,active_learning,alignment}.py` (pure numerics) · thin `QThread` wrappers + session in `gui/ml/` · panels in `gui/components/ml_tabs/` · shell `gui/suites/ml_suite.py`
- **PEEM:** `core/peem/{bg,engine,roi,sumrule}.py` · service `gui/services/peem_service.py` · panel/suite under `gui/`
- **DFT / ARPES:** `core/dft/`, `core/arpes/` (+ top-level engines where still present)

## Still optional later

- Rename `gui/ml/maestroai_*.py` workers to clearer names
- Split fat Crystal suite into `components/crystal_tabs/`
