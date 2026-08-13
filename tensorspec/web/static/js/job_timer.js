/* Elapsed clock + labeled ETA for long suite jobs (DFT Queue, ARPES Simulate).
   Heuristic formulas match tensorspec/core/jobs/eta_heuristic.py. */

const JobTimer = (() => {
    const STORAGE_KEY = "tensorspec.jobTimes.v1";
    const timers = new WeakMap();

    function formatElapsed(seconds) {
        const total = Math.max(0, Math.floor(Number(seconds) || 0));
        const hours = Math.floor(total / 3600);
        const minutes = Math.floor((total % 3600) / 60);
        const secs = total % 60;
        if (hours) {
            return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
        }
        return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    }

    function formatEstimate(seconds) {
        const total = Math.max(0, Math.floor(Number(seconds) || 0));
        if (total >= 90) {
            const minutes = Math.max(1, Math.round(total / 60));
            return `~${minutes} min`;
        }
        return `~${total} s`;
    }

    function estimateDftSeconds({ backend, nbnd, kx, ky, kz, soc, ranks }) {
        const base = backend === "einstein_ssh" ? 20 * 60 : 60 * 60;
        const seconds =
            base
            * ((Number(nbnd) || 12) / 162)
            * (((Number(kx) || 1) * (Number(ky) || 1) * (Number(kz) || 1)) / 216)
            * (soc ? 2 : 1)
            * (8 / Math.max(Number(ranks) || 1, 1));
        return Math.max(120, Math.min(12 * 3600, Math.floor(seconds)));
    }

    function estimateArpesSeconds({ nEnergy, nKx, nKy }) {
        const voxels = Math.max(1, (Number(nEnergy) || 1) * (Number(nKx) || 1) * (Number(nKy) || 1));
        const seconds = 180 * (voxels / (48 * 64 * 64));
        return Math.max(30, Math.min(2 * 3600, Math.floor(seconds)));
    }

    function nbndBin(nbnd) {
        const n = Number(nbnd) || 0;
        if (n <= 20) return 12;
        if (n <= 150) return 100;
        if (n <= 300) return 200;
        return 400;
    }

    function kprodBin(kx, ky, kz) {
        const p = (Number(kx) || 1) * (Number(ky) || 1) * (Number(kz) || 1);
        if (p <= 8) return 8;
        if (p <= 64) return 64;
        if (p <= 216) return 216;
        return 512;
    }

    function voxelBin(nEnergy, nKx, nKy) {
        const v = Math.max(1, (Number(nEnergy) || 1) * (Number(nKx) || 1) * (Number(nKy) || 1));
        if (v <= 1e4) return 10000;
        if (v <= 1e5) return 100000;
        if (v <= 2e5) return 200000;
        return 500000;
    }

    function dftKey({ backend, soc, nbnd, kx, ky, kz }) {
        return `dft:qe:${backend || "local"}:${soc ? 1 : 0}:${nbndBin(nbnd)}:${kprodBin(kx, ky, kz)}`;
    }

    function arpesKey({ model, nEnergy, nKx, nKy }) {
        return `arpes:sim:${model || "A"}:${voxelBin(nEnergy, nKx, nKy)}`;
    }

    function readStore() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") || {};
        } catch (err) {
            return {};
        }
    }

    function writeStore(map) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
        } catch (err) {
            /* quota / private mode */
        }
    }

    function lookupLast(key) {
        const raw = readStore()[key];
        const n = Number(raw);
        return Number.isFinite(n) && n > 0 ? n : null;
    }

    function remember(key, seconds) {
        if (!key) return;
        const n = Math.max(1, Math.floor(Number(seconds) || 0));
        const map = readStore();
        map[key] = n;
        writeStore(map);
    }

    function clearTimer(el) {
        const prev = timers.get(el);
        if (prev?.interval) clearInterval(prev.interval);
        timers.delete(el);
    }

    function paintRunning(el, state) {
        const elapsed = (Date.now() - state.t0) / 1000;
        el.hidden = false;
        el.textContent =
            `elapsed ${formatElapsed(elapsed)} · est. ${formatEstimate(state.estimateSeconds)} (${state.estimateSource})`;
    }

    function start(el, { estimateSeconds, estimateSource } = {}) {
        if (!el) return;
        clearTimer(el);
        const state = {
            t0: Date.now(),
            estimateSeconds: Number(estimateSeconds) || 120,
            estimateSource: estimateSource === "last run" ? "last run" : "heuristic",
            interval: null,
        };
        paintRunning(el, state);
        state.interval = setInterval(() => paintRunning(el, state), 1000);
        timers.set(el, state);
    }

    function elapsedSeconds(el) {
        const state = el && timers.get(el);
        if (!state) return 0;
        return Math.max(0, Math.floor((Date.now() - state.t0) / 1000));
    }

    function stop(el, terminalStatus) {
        if (!el) return;
        const secs = elapsedSeconds(el);
        clearTimer(el);
        const clock = formatElapsed(secs);
        let line = `finished in ${clock}`;
        if (terminalStatus === "failed") line = `finished in ${clock} (failed)`;
        if (terminalStatus === "cancelled") line = `cancelled at ${clock}`;
        el.hidden = false;
        el.textContent = line;
    }

    return {
        formatElapsed,
        formatEstimate,
        estimateDftSeconds,
        estimateArpesSeconds,
        dftKey,
        arpesKey,
        lookupLast,
        remember,
        start,
        stop,
        elapsedSeconds,
    };
})();

window.JobTimer = JobTimer;
