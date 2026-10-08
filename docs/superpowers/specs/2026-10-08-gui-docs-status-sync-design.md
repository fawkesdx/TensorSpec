# TensorSpec_GUI docs status sync

**Date:** 2026-10-08  
**Branch:** `TensorSpec_GUI`  
**Approach:** A (status sync) — approved in chat

## Goal

Align `README.md`, `roadmap.md`, `sandy_rule.md`, and `requirements.txt` with the current Qt GUI tree. No feature code changes.

## Scope

| File | Change |
|------|--------|
| `README.md` | GUI identity, install, launch, suite status table, remote GPU link |
| `roadmap.md` | Checkbox sync for PEEM / XAS / Transport / partial ARPES viewer+MAESTRO / ML; no full rewrite |
| `sandy_rule.md` | §7 folder blueprint only (`core/peem/`, services, ml, maestro loaders) |
| `requirements.txt` | Drop PyQt6 pins; light grouping comments; keep versions otherwise |

## Out of scope

- HTML / Einstein docs
- Implementing `transport_engine`
- Rewriting Crystal/DFT roadmap history
- Force-push or merge to `main`

## Suite status (source of truth for this edit)

| Suite | Status |
|-------|--------|
| Crystal, DFT, ARPES simulation | Live |
| PEEM | Live (core + Qt panel) |
| XAS / XMCD 1D | Live (shared PEEM BG/sum-rule) |
| Transport | UI shell / demo only |
| ML | Live (suite + DataViewer domain overlay) |
| DataViewer / MAESTRO kinds | Partial live |

## Verification

- Spec matches Approach A design approved 2026-10-08
- After edit: no inventing finished beamline loaders; Transport remains pending engine
