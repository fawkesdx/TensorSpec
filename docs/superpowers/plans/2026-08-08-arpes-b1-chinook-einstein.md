# Option B1 + Chinook Einstein Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install chinook on Einstein `TensorSpec_env` and unlock Option B1 for remote CLI + web Queue Einstein backend.

**Architecture:** SSH pip install chinook; extend `run_arpes_me_a.py` to accept A|B1; remove web 422 gate and UI B1 hide; stop forcing model A in remote job JSON / results.

**Tech Stack:** pip/chinook, FastAPI ARPES router, existing `remote_arpes_me.sh`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-arpes-b1-chinook-einstein-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- Keep filename `scripts/run_arpes_me_a.py`
- Do not change `remote_arpes_me.sh` behavior except docs if needed
- Exit 6 when chinook required but missing
- Option A + Simple Scalar remains chinook-free
- No live SSH in CI unit tests; Einstein install/smoke is controller Task 1 / Task 5

## File map

| File | Role |
|------|------|
| Einstein `TensorSpec_env` | `pip install chinook` |
| `scripts/run_arpes_me_a.py` | Accept A\|B1; pass model through |
| `scripts/README-remote-arpes-me.md` | Chinook / B1 docs |
| `tensorspec/web/server/routers/arpes.py` | Drop B1 refuse; keep request model |
| `tensorspec/web/static/js/arpes_suite.js` | Re-enable B1 on Einstein |
| `tensorspec/web/templates/suites/arpes_suite.html` | Hint text |
| `tests/test_run_arpes_me_a.py` | B1 accepted as model; unknown rejected |
| `tests/test_arpes_einstein_backend.py` | Remove 422 B1 test; assert model passthrough |

---

### Task 1: Install chinook on Einstein (controller / shell agent)

**Files:** none in git (env only); optionally note in README in Task 3

- [ ] **Step 1: Install**

```bash
ssh -o BatchMode=yes einstein 'cd ~/TensorSpec && ./TensorSpec_env/bin/pip install chinook'
```

- [ ] **Step 2: Verify import**

```bash
ssh -o BatchMode=yes einstein 'cd ~/TensorSpec && PYTHONPATH=. ./TensorSpec_env/bin/python -c "import chinook; print(\"chinook_ok\", chinook.__file__)"'
```

Expected: prints `chinook_ok` and a path under `TensorSpec_env`.

- [ ] **Step 3: Ledger note** — record pip success in SDD progress (no git commit required for env-only). If pip fails, STOP and report.

---

### Task 2: CLI accept A|B1

**Files:**
- Modify: `scripts/run_arpes_me_a.py`
- Modify: `tests/test_run_arpes_me_a.py`

**Interfaces:**
- `model` in `{"A","B1"}` else exit 2
- `ARPESEngineRouter().run_simulation(model, …)` with actual model
- `meta.json` `"model": model`
- ImportError on mesh → 6; B1 requiring chinook surfaces as ImportError/RuntimeError → prefer map chinook missing to 6 when message mentions chinook

- [ ] **Step 1: Update failing/adjust tests**

Replace `test_rejects_b1` with:

```python
def test_rejects_unknown_model(self):
    with TemporaryDirectory() as tmp:
        job = Path(tmp)
        _write_si_job(job, model="B2")  # or patch request.json model to "X"
        # If _write_si_job only allows A/B1, write request manually with model "X"
        ...
        self.assertEqual(r.returncode, 2)

def test_accepts_b1_model_field(self):
    """B1 is a valid model string; may exit 6/4 if chinook missing in CI."""
    with TemporaryDirectory() as tmp:
        job = Path(tmp)
        _write_si_job(job, model="B1", mesh=4, steps=4)
        r = subprocess.run([PY, str(SCRIPT), str(job)], capture_output=True, text=True)
        # Valid model: not the old "Option A only" validation
        self.assertNotIn("Option A only", r.stderr + r.stdout)
        self.assertIn(r.returncode, (0, 4, 6))
```

Keep `test_rejects_b1` removed. Keep tiny Option A test.

- [ ] **Step 2: Run — expect FAIL** (old code still rejects B1 with exit 2)

- [ ] **Step 3: Implement CLI**

In `run_arpes_me_a.py`:

```python
model = str(req.get("model", "A"))
if model not in ("A", "B1"):
    raise ValueError(f"model must be 'A' or 'B1' (got {model!r})")
...
results = arpes.run_simulation(model, band_data, _experiment_kwargs(req))
...
meta = { "model": model, ... }
```

Update module docstring: Option A/B1 entrypoint.

Optional: before B1 run, if chinook import fails, return 6 early:

```python
if model == "B1":
    try:
        import chinook  # noqa: F401
    except ImportError as e:
        _log(job_dir, f"[arpes] missing dependency: {e}")
        return 6
```

- [ ] **Step 4: Tests PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/run_arpes_me_a.py tests/test_run_arpes_me_a.py
git commit -m "$(cat <<'EOF'
feat: remote ARPES runner accepts Option B1

Require chinook for B1; keep A|B1 validation and meta.model.
EOF
)"
```

---

### Task 3: Web unlock + README

**Files:**
- Modify: `tensorspec/web/server/routers/arpes.py`
- Modify: `tensorspec/web/static/js/arpes_suite.js`
- Modify: `tensorspec/web/templates/suites/arpes_suite.html` (hint)
- Modify: `scripts/README-remote-arpes-me.md`
- Modify: `tests/test_arpes_einstein_backend.py`

**Interfaces:**
- Remove call to `_refuse_b1_on_einstein` from `queue_simulation`; delete helper
- `_request_json_for_remote`: do **not** force `model` to `"A"` — use `request.model_dump()` as-is (or only normalize unrelated fields)
- Einstein worker `job.result["model"]` = `request.model`
- `syncArpesBackendUi`: no longer disable/hide B1 (function can become no-op or only update hint); remove force-to-A
- Tests: replace `test_b1_einstein_returns_422` with `test_b1_einstein_allowed` (calling deleted helper should not exist — assert `_refuse_b1_on_einstein` gone OR that queue path doesn't 422). Simplest: delete 422 test; add assert `_request_json_for_remote` keeps `model=B1`

- [ ] **Step 1: Failing tests**

```python
def test_request_json_keeps_b1(self):
    req = _tiny_request(backend="einstein_ssh", model="B1")
    data = arpes_router._request_json_for_remote(req)
    self.assertEqual(data["model"], "B1")
```

Remove `test_b1_einstein_returns_422`.

- [ ] **Step 2: Implement**

```python
def _request_json_for_remote(request: ArpesSimRequest) -> dict:
    return request.model_dump()
```

In worker result dict: `"model": request.model`

Delete `_refuse_b1_on_einstein` and its call.

JS:

```javascript
function syncArpesBackendUi() {
    // B1 allowed on Einstein when chinook is installed remotely.
}
```

Or remove calls entirely — prefer keep empty function + listeners for future hints.

HTML hint: change to say Option A/B1 on Einstein; B1 needs chinook in TensorSpec_env.

README: add Chinook section — required for B1 and Slater-Koster; install command; A+Scalar without chinook OK; model A|B1 in request.json.

- [ ] **Step 3: Tests PASS**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_arpes_einstein_backend tests.test_run_arpes_me_a -v
```

- [ ] **Step 4: Commit**

```bash
git add tensorspec/web/server/routers/arpes.py tensorspec/web/static/js/arpes_suite.js \
  tensorspec/web/templates/suites/arpes_suite.html scripts/README-remote-arpes-me.md \
  tests/test_arpes_einstein_backend.py
git commit -m "$(cat <<'EOF'
feat: unlock ARPES Option B1 on Einstein Queue

Drop B1 422 gate; pass model through remote job JSON and UI.
EOF
)"
```

---

### Task 4: Push + Einstein pull + smoke

- [ ] Push `HTML_einstein_app`
- [ ] Einstein `git pull`
- [ ] CLI smoke B1 tiny job:

```bash
# prepare job with model B1, mesh 4, steps 4
./scripts/remote_arpes_me.sh /tmp/arpes_me_b1_smoke
```

Expected: exit 0, `intensity.npz` present. If B1 physics fails for Si, document and fall back to proving import + model path with log showing B1 started (not exit 2).

---

## Spec coverage

| Spec item | Task |
|-----------|------|
| pip chinook | 1 |
| CLI A\|B1 | 2 |
| Exit 6 | 2 |
| Web unlock | 3 |
| UI | 3 |
| README | 3 |
| Smoke | 4 |

## Self-review

- No rename of script.
- Model forced `"A"` removed in both CLI meta and web JSON/result.
- Old 422 test deleted, not left asserting obsolete behavior.
