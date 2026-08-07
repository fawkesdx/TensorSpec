/* DFT Suite controller: tight-binding bands and the QE pipeline.

 * Band structures are solved synchronously. QE runs are queued: the server
 * builds argv lists from its allowlist, never from typed executable paths.
 */

import { BandPlot } from "/static/js/viewers/band_plot.js";

const el = (id) => document.getElementById(id);

const dom = {
    structures: el("dft-structures"),
    refresh: el("dft-refresh"),
    calculate: el("dft-calculate"),
    status: el("dft-status"),
    badge: el("dft-badge"),
    viewport: el("dft-viewport"),
    placeholder: el("dft-placeholder"),

    pathMode: el("tb-path"),
    coords: el("tb-coords"),
    labels: el("tb-labels"),
    points: el("tb-pts"),
    hamiltonian: el("tb-ham"),
    soc: el("tb-soc"),
    socStrength: el("tb-soc-val"),
    onsite: el("tb-onsite"),
    shiftS: el("tb-shift-s"),
    shiftP: el("tb-shift-p"),
    shiftD: el("tb-shift-d"),
    eMin: el("tb-emin"),
    eMax: el("tb-emax"),
    shells: el("tb-shells"),

    qeCutoff: el("qe-cutoff"),
    qeSoc: el("qe-soc"),
    qeNbnd: el("qe-nbnd"),
    qeWanMode: el("qe-wanmode"),
    qeKx: el("qe-kx"),
    qeKy: el("qe-ky"),
    qeKz: el("qe-kz"),
    qeRunName: el("qe-runname"),
    qeMpi: el("qe-mpi"),
    qeRanks: el("qe-ranks"),
    qeGenerate: el("qe-generate"),
    qeBundle: el("qe-bundle"),
    qeQueue: el("qe-queue"),
    qeCancel: el("qe-cancel"),
    qeStatus: el("qe-status"),
    qeLog: el("qe-log"),

    statBands: el("dft-stat-bands"),
    statTime: el("dft-stat-time"),
};

const PATH_VALUES = {
    "Auto-Detect BZ Path (PyMatgen)": "auto",
    "Arbitrary Custom Path": "custom",
    "Hexagonal (Template)": "hexagonal",
    "Rectangular / Orthorhombic (Template)": "rectangular",
    "Square / Tetragonal (Template)": "square",
};

let plot = null;
let structures = [];
let activeJobId = null;
let logSocket = null;
let maxMpiRanks = 8;

function setStatus(message, isError = false) {
    dom.status.textContent = message;
    dom.status.style.color = isError ? "#ff6b6b" : "";
}

function setQeStatus(message, isError = false) {
    dom.qeStatus.textContent = message;
    dom.qeStatus.style.color = isError ? "#ff6b6b" : "";
}

function ensurePlot() {
    if (plot) return plot;
    if (dom.placeholder) dom.placeholder.remove();
    plot = new BandPlot(dom.viewport);
    return plot;
}

function selected() {
    return structures.find((s) => s.name === dom.structures.value) || null;
}

function renderShells() {
    const structure = selected();
    dom.shells.innerHTML = "";
    if (!structure) return;

    structure.shell_keys.forEach((key, index) => {
        const value = structure.default_hoppings[index] ?? 0;
        const row = document.createElement("div");
        row.className = "form-row";
        row.innerHTML = `
            <label>${key}:</label>
            <div class="inline">
                <input class="field field--num" data-hop type="number" step="0.1" min="-10" max="10" value="${value}">
                <label>Max &#197;:</label>
                <input class="field field--num" data-cut type="number" step="0.1" min="0" max="15"
                       value="${[1.6, 2.6, 3.1, 4.5][index] ?? 4.5}">
            </div>`;
        dom.shells.appendChild(row);
    });
}

async function refreshStructures() {
    try {
        structures = await TensorSpecAPI.dftStructures();
        dom.structures.innerHTML = "";

        if (!structures.length) {
            dom.structures.innerHTML = '<option value="">No structures available</option>';
            dom.calculate.disabled = true;
            setStatus("Load a CIF in the Crystal Suite first.");
            renderShells();
            return;
        }

        structures.forEach((s) => {
            const option = document.createElement("option");
            option.value = s.name;
            option.textContent = `${s.name} (${s.formula}, ${s.n_sites} sites)`;
            dom.structures.appendChild(option);
        });

        dom.calculate.disabled = false;
        renderShells();
        setStatus(`${structures.length} structure(s) available`);
    } catch (err) {
        setStatus(err.message, true);
    }
}

function readParameters() {
    const hoppings = [...dom.shells.querySelectorAll("[data-hop]")].map((i) => Number(i.value) || 0);
    const cutoffs = [...dom.shells.querySelectorAll("[data-cut]")].map((i) => Number(i.value) || 0);

    return {
        path_mode: PATH_VALUES[dom.pathMode.value] || dom.pathMode.value || "auto",
        custom_coords: dom.coords.value,
        custom_labels: dom.labels.value,
        points_per_segment: Number(dom.points.value) || 100,
        hoppings: hoppings.length ? hoppings : [2.7, 0, 0, -0.3],
        cutoffs: cutoffs.length ? cutoffs : [1.6, 2.6, 3.1, 4.5],
        onsite_e: Number(dom.onsite.value) || 0,
        shift_s: Number(dom.shiftS.value) || 0,
        shift_p: Number(dom.shiftP.value) || 0,
        shift_d: Number(dom.shiftD.value) || 0,
        use_soc: dom.soc.checked,
        soc_strength: Number(dom.socStrength.value) || 0.5,
        tb_mode: dom.hamiltonian.value,
    };
}

function readQeParameters() {
    return {
        run_name: dom.qeRunName.value.trim() || "run_01",
        ecutwfc: Number(dom.qeCutoff.value) || 60,
        nbnd: Number(dom.qeNbnd.value) || 12,
        kx: Number(dom.qeKx.value) || 6,
        ky: Number(dom.qeKy.value) || 6,
        kz: Number(dom.qeKz.value) || 6,
        use_soc: dom.qeSoc.checked,
        mlwf_mode: dom.qeWanMode.value === "mlwf",
        use_mpi: dom.qeMpi.checked,
        mpi_ranks: Number(dom.qeRanks.value) || 4,
    };
}

async function calculate() {
    const structure = selected();
    if (!structure) return;

    dom.calculate.disabled = true;
    setStatus(`Solving ${structure.formula}\u2026`);
    try {
        const result = await TensorSpecAPI.dftBands(structure.name, readParameters());

        const view = ensurePlot();
        view.setResult(result);
        view.setRange(Number(dom.eMin.value) || -6, Number(dom.eMax.value) || 6);

        dom.statBands.textContent = `${result.n_bands} bands, ${result.n_kpoints} k-points`;
        dom.statTime.textContent = `solved in ${result.elapsed_seconds}s`;
        dom.badge.textContent = result.node_labels.join(" \u2192 ") || "computed";
        setStatus(`Pushed to workspace as ${result.name}`);
    } catch (err) {
        setStatus(err.message, true);
    } finally {
        dom.calculate.disabled = false;
    }
}

function appendLog(line) {
    dom.qeLog.hidden = false;
    dom.qeLog.textContent += `${line}\n`;
    dom.qeLog.scrollTop = dom.qeLog.scrollHeight;
}

function closeLogSocket() {
    if (logSocket) {
        logSocket.close();
        logSocket = null;
    }
}

function watchJob(jobId) {
    closeLogSocket();
    activeJobId = jobId;
    dom.qeCancel.disabled = false;
    dom.qeLog.textContent = "";
    dom.qeLog.hidden = false;

    const protocol = location.protocol === "https:" ? "wss" : "ws";
    logSocket = new WebSocket(`${protocol}://${location.host}/api/dft/jobs/${encodeURIComponent(jobId)}/logs`);
    logSocket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "log") appendLog(message.line);
        if (message.type === "status") {
            const job = message.job;
            setQeStatus(`${job.run_name}: ${job.status}` + (job.error ? ` — ${job.error}` : ""));
            if (["succeeded", "failed", "cancelled"].includes(job.status)) {
                dom.qeCancel.disabled = true;
                activeJobId = null;
            }
        }
        if (message.type === "error") setQeStatus(message.message, true);
    };
    logSocket.onerror = () => setQeStatus("Log stream disconnected", true);
}

async function generateInputs() {
    const structure = selected();
    if (!structure) {
        setQeStatus("Select a crystal first.", true);
        return;
    }
    dom.qeGenerate.disabled = true;
    try {
        const result = await TensorSpecAPI.qeGenerate(structure.name, readQeParameters());
        setQeStatus(
            `Wrote ${result.files.join(", ")} → ${result.run_dir} `
            + `(MPI ranks capped at ${result.mpi_ranks_capped}/${result.max_mpi_ranks})`
        );
    } catch (err) {
        setQeStatus(err.message, true);
    } finally {
        dom.qeGenerate.disabled = false;
    }
}

async function downloadBundle() {
    const structure = selected();
    if (!structure) {
        setQeStatus("Select a crystal first.", true);
        return;
    }
    dom.qeBundle.disabled = true;
    try {
        const params = readQeParameters();
        const blob = await TensorSpecAPI.qeBundle(structure.name, params);
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${params.run_name}.zip`;
        link.click();
        URL.revokeObjectURL(url);
        setQeStatus(`Downloaded ${params.run_name}.zip for HPC`);
    } catch (err) {
        setQeStatus(err.message, true);
    } finally {
        dom.qeBundle.disabled = false;
    }
}

async function queueRun() {
    const structure = selected();
    if (!structure) {
        setQeStatus("Select a crystal first.", true);
        return;
    }
    dom.qeQueue.disabled = true;
    try {
        const job = await TensorSpecAPI.qeQueue(structure.name, readQeParameters());
        setQeStatus(`Queued ${job.run_name} (${job.job_id})`);
        watchJob(job.job_id);
    } catch (err) {
        setQeStatus(err.message, true);
    } finally {
        dom.qeQueue.disabled = false;
    }
}

async function cancelRun() {
    if (!activeJobId) return;
    try {
        const job = await TensorSpecAPI.qeCancel(activeJobId);
        setQeStatus(`${job.run_name}: ${job.status}`);
    } catch (err) {
        setQeStatus(err.message, true);
    }
}

async function refreshSolvers() {
    try {
        const info = await TensorSpecAPI.dftSolvers();
        maxMpiRanks = info.max_mpi_ranks || 8;
        dom.qeRanks.max = maxMpiRanks;
        if (Number(dom.qeRanks.value) > maxMpiRanks) dom.qeRanks.value = maxMpiRanks;

        if (info.available) {
            setQeStatus(
                `Solvers ready — max ${maxMpiRanks} MPI ranks`
                + (info.mpirun ? "" : " (mpirun not found; runs will be serial)")
            );
            dom.qeQueue.disabled = false;
        } else {
            setQeStatus(`Solvers unavailable: ${info.detail || "check server config"}`, true);
            dom.qeQueue.disabled = true;
        }
    } catch (err) {
        setQeStatus(err.message, true);
        dom.qeQueue.disabled = true;
    }
}

dom.refresh.addEventListener("click", refreshStructures);
dom.structures.addEventListener("change", renderShells);
dom.calculate.addEventListener("click", calculate);
dom.qeGenerate.addEventListener("click", generateInputs);
dom.qeBundle.addEventListener("click", downloadBundle);
dom.qeQueue.addEventListener("click", queueRun);
dom.qeCancel.addEventListener("click", cancelRun);

dom.soc.addEventListener("change", () => {
    dom.socStrength.disabled = !dom.soc.checked;
});

[dom.eMin, dom.eMax].forEach((input) =>
    input.addEventListener("change", () => {
        if (plot) plot.setRange(Number(dom.eMin.value) || -6, Number(dom.eMax.value) || 6);
    })
);

function syncPathFields() {
    const isCustom = PATH_VALUES[dom.pathMode.value] === "custom";
    dom.coords.disabled = !isCustom;
    dom.labels.disabled = !isCustom;
}
dom.pathMode.addEventListener("change", syncPathFields);
syncPathFields();

refreshStructures();
refreshSolvers();
