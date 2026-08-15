# DFT job list + reconnect (HTML_einstein_app)

**Date:** 2026-08-15  
**Branch:** `HTML_einstein_app`  
**Problem:** Leave DFT suite → UI forgets job; uvicorn `--reload` kills in-memory queue.

## Ship

1. Einstein uvicorn **without** `--reload`
2. `sessionStorage` + DFT page resume via log WebSocket
3. `GET /api/dft/jobs` — session jobs, active first

## Out

- Persist queue across process death (disk)
- Keep `pw.x` alive across uvicorn restart
