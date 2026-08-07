/* DFT Suite controller: tight-binding band structures.
 *
 * Collects parameters, asks the server to solve, and hands the eigenvalues to
 * the plot. The Quantum ESPRESSO panel stays inert here on purpose: running
 * solvers needs the job queue and the executable allowlist, not a fetch call.
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

function setStatus(message, isError = false) {
    dom.status.textContent = message;
    dom.status.style.color = isError ? "#ff6b6b" : "";
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

/* The four t boxes are anonymous until a material defines its shells. */
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

dom.refresh.addEventListener("click", refreshStructures);
dom.structures.addEventListener("change", renderShells);
dom.calculate.addEventListener("click", calculate);

dom.soc.addEventListener("change", () => {
    dom.socStrength.disabled = !dom.soc.checked;
});

[dom.eMin, dom.eMax].forEach((input) =>
    input.addEventListener("change", () => {
        if (plot) plot.setRange(Number(dom.eMin.value) || -6, Number(dom.eMax.value) || 6);
    })
);

/* Custom coordinate fields only matter for the arbitrary path. */
function syncPathFields() {
    const isCustom = PATH_VALUES[dom.pathMode.value] === "custom";
    dom.coords.disabled = !isCustom;
    dom.labels.disabled = !isCustom;
}
dom.pathMode.addEventListener("change", syncPathFields);
syncPathFields();

refreshStructures();
