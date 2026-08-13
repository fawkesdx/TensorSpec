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
    pathNote: el("dft-path-note"),
    kgrid: el("tb-kgrid"),
    isoe: el("tb-isoe"),
    gapFid: el("dft-gap-fid"),
    gapPredict: el("dft-gap-predict"),
    gapStatus: el("dft-gap-status"),
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
    fat: el("tb-fat"),
    w90File: el("tb-w90-file"),
    w90Load: el("tb-w90-load"),
    w90Use: el("tb-w90-use"),
    w90Overlay: el("tb-w90-overlay"),
    w90Status: el("tb-w90-status"),

    qeCutoff: el("qe-cutoff"),
    qeXc: el("qe-xc"),
    qeSoc: el("qe-soc"),
    qeNbnd: el("qe-nbnd"),
    qeWanMode: el("qe-wanmode"),
    qeKx: el("qe-kx"),
    qeKy: el("qe-ky"),
    qeKz: el("qe-kz"),
    qeSlabMode: el("qe-slab-mode"),
    qeRunName: el("qe-runname"),
    qeMpi: el("qe-mpi"),
    qeRanks: el("qe-ranks"),
    qeGenerate: el("qe-generate"),
    qeBundle: el("qe-bundle"),
    qeBackend: el("qe-backend"),
    qeQueue: el("qe-queue"),
    qeCancel: el("qe-cancel"),
    qeStatus: el("qe-status"),
    qeLog: el("qe-log"),

    qeSlabPreset: el("qe-slab-preset"),
    qeSlabH: el("qe-slab-h"),
    qeSlabK: el("qe-slab-k"),
    qeSlabL: el("qe-slab-l"),
    qeSlabLayers: el("qe-slab-layers"),
    qeSlabVac: el("qe-slab-vac"),
    qeSlabStore: el("qe-slab-store"),
    qeSlabPrepare: el("qe-slab-prepare"),
    qeSlabStatus: el("qe-slab-status"),

    statBands: el("dft-stat-bands"),
    statTime: el("dft-stat-time"),
};

const PATH_VALUES = {
    "Auto-Detect BZ Path (PyMatgen)": "auto",
    "Primitive hex reference (folded into supercell)": "primitive_hex_ref",
    "Unfold hex (spectral weight)": "unfold_hex",
    "Arbitrary Custom Path": "custom",
    "Hexagonal (Template)": "hexagonal",
    "Rectangular / Orthorhombic (Template)": "rectangular",
    "Square / Tetragonal (Template)": "square",
};

let plot = null;
let heatmapCanvas = null;
let structures = [];
let activeJobId = null;
let logSocket = null;
let maxMpiRanks = 20;
let lastSolversInfo = null;
let lastBandResult = null;
let lastCrystalName = null;

function setStatus(message, isError = false) {
    dom.status.textContent = message;
    dom.status.style.color = isError ? "#ff6b6b" : "";
}

function setQeStatus(message, isError = false) {
    if (!dom.qeStatus) return;
    dom.qeStatus.textContent = message;
    dom.qeStatus.style.color = isError ? "#ff6b6b" : "";
}

function isIsoenergyMode() {
    return dom.kgrid && dom.kgrid.value === "isoenergy";
}

function syncKgridMode() {
    const iso = isIsoenergyMode();
    if (dom.isoe) dom.isoe.disabled = !iso;
    if (dom.pathMode) dom.pathMode.disabled = iso;
    if (dom.coords) dom.coords.disabled = iso || PATH_VALUES[dom.pathMode.value] !== "custom";
    if (dom.labels) dom.labels.disabled = iso || PATH_VALUES[dom.pathMode.value] !== "custom";
    if (dom.points) dom.points.disabled = iso;
    if (dom.fat) dom.fat.disabled = iso || !lastBandResult;
}

function ensurePlot() {
    hideHeatmap();
    if (plot) {
        if (plot.canvas) plot.canvas.style.display = "block";
        return plot;
    }
    if (dom.placeholder) dom.placeholder.remove();
    plot = new BandPlot(dom.viewport);
    return plot;
}

function hideHeatmap() {
    if (heatmapCanvas) {
        heatmapCanvas.hidden = true;
    }
}

function ensureHeatmap() {
    if (dom.placeholder) dom.placeholder.remove();
    if (plot && plot.canvas) {
        plot.canvas.style.display = "none";
    }
    if (!heatmapCanvas) {
        heatmapCanvas = document.createElement("canvas");
        heatmapCanvas.id = "dft-isoenergy-heatmap";
        heatmapCanvas.style.width = "100%";
        heatmapCanvas.style.height = "100%";
        heatmapCanvas.style.display = "block";
        heatmapCanvas.style.background = "#111";
        dom.viewport.appendChild(heatmapCanvas);
    }
    heatmapCanvas.hidden = false;
    return heatmapCanvas;
}

function renderIsoenergyHeatmap(result) {
    const canvas = ensureHeatmap();
    const intensity = result.intensity;
    const nx = intensity.length;
    const ny = intensity[0] ? intensity[0].length : 0;
    if (!nx || !ny) return;

    const padL = 56;
    const padR = 16;
    const padT = 16;
    const padB = 44;
    const cssW = canvas.clientWidth || dom.viewport.clientWidth || 640;
    const cssH = canvas.clientHeight || dom.viewport.clientHeight || 480;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(cssW * dpr));
    canvas.height = Math.max(1, Math.floor(cssH * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, cssW, cssH);

    let vmax = 0;
    for (let ix = 0; ix < nx; ix++) {
        for (let iy = 0; iy < ny; iy++) {
            const v = intensity[ix][iy];
            if (v > vmax) vmax = v;
        }
    }
    if (vmax <= 0) vmax = 1;

    const plotW = Math.max(1, cssW - padL - padR);
    const plotH = Math.max(1, cssH - padT - padB);
    const cellW = plotW / nx;
    const cellH = plotH / ny;

    for (let ix = 0; ix < nx; ix++) {
        for (let iy = 0; iy < ny; iy++) {
            const t = intensity[ix][iy] / vmax;
            const r = Math.round(20 + 220 * t);
            const g = Math.round(30 + 180 * t);
            const b = Math.round(80 + 100 * (1 - t));
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            // iy=0 at bottom (ky_min)
            const x = padL + ix * cellW;
            const y = padT + (ny - 1 - iy) * cellH;
            ctx.fillRect(x, y, Math.ceil(cellW) + 0.5, Math.ceil(cellH) + 0.5);
        }
    }

    ctx.strokeStyle = "#888";
    ctx.strokeRect(padL, padT, plotW, plotH);
    ctx.fillStyle = "#ddd";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    const kx0 = result.kx[0];
    const kx1 = result.kx[result.kx.length - 1];
    const ky0 = result.ky[0];
    const ky1 = result.ky[result.ky.length - 1];
    ctx.fillText(`kx (${kx0.toFixed(2)} … ${kx1.toFixed(2)})`, padL + plotW / 2, cssH - 12);
    ctx.save();
    ctx.translate(14, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`ky (${ky0.toFixed(2)} … ${ky1.toFixed(2)})`, 0, 0);
    ctx.restore();
    ctx.textAlign = "left";
    ctx.fillText(`E = ${result.energy.toFixed(3)} eV  σ = ${result.smear}`, padL, 12);
}

function selected() {
    return structures.find((s) => s.name === dom.structures.value) || null;
}

function shellKind(label) {
    const base = label.replace(/_up$|_dn$/, "");
    if (base.endsWith("_s")) return "s";
    if (/_(pz|px|py)$/.test(base)) return "p";
    if (/_(dz2|dxz|dyz|dx2-y2|dxy)$/.test(base)) return "d";
    return null;
}

function elementFromLabel(label) {
    const i = label.indexOf("_");
    return i > 0 ? label.slice(0, i) : null;
}

function populateFatOptions(orbitalLabels) {
    if (!dom.fat) return;
    const labels = orbitalLabels || [];
    dom.fat.innerHTML = "";
    const none = document.createElement("option");
    none.value = "none";
    none.textContent = "None (Standard Lines)";
    dom.fat.appendChild(none);

    if (!labels.length) {
        dom.fat.disabled = true;
        return;
    }

    const shells = new Set();
    const elements = new Set();
    labels.forEach((lab) => {
        const sk = shellKind(lab);
        if (sk) shells.add(sk);
        const elSym = elementFromLabel(lab);
        if (elSym) elements.add(elSym);
    });

    if (shells.size) {
        const group = document.createElement("optgroup");
        group.label = "Shells";
        ["s", "p", "d"].forEach((sk) => {
            if (!shells.has(sk)) return;
            const opt = document.createElement("option");
            opt.value = `shell:${sk}`;
            opt.textContent = `All ${sk}`;
            group.appendChild(opt);
        });
        dom.fat.appendChild(group);
    }

    if (elements.size) {
        const group = document.createElement("optgroup");
        group.label = "Elements";
        [...elements].sort().forEach((elSym) => {
            const opt = document.createElement("option");
            opt.value = `element:${elSym}`;
            opt.textContent = elSym;
            group.appendChild(opt);
        });
        dom.fat.appendChild(group);
    }

    const orbGroup = document.createElement("optgroup");
    orbGroup.label = "Orbitals";
    [...new Set(labels)].forEach((lab) => {
        const opt = document.createElement("option");
        opt.value = `orbital:${lab}`;
        opt.textContent = lab;
        orbGroup.appendChild(opt);
    });
    dom.fat.appendChild(orbGroup);
    dom.fat.disabled = false;
    dom.fat.value = "none";
}

async function applyFatTarget() {
    if (!lastBandResult || !lastCrystalName || !dom.fat) return;
    const target = dom.fat.value || "none";
    if (target === "none") {
        lastBandResult = { ...lastBandResult, fat_weights: null, fat_target: "none" };
        ensurePlot().setResult(lastBandResult);
        setStatus("Fat bands cleared");
        return;
    }
    try {
        const fat = await TensorSpecAPI.dftFatBands(lastCrystalName, target);
        lastBandResult = {
            ...lastBandResult,
            fat_weights: fat.fat_weights,
            fat_target: fat.fat_target,
            fat_n_orbitals: fat.fat_n_orbitals,
        };
        ensurePlot().setResult(lastBandResult);
        setStatus(
            fat.fat_weights
                ? `Fat ${fat.fat_target} (${fat.fat_n_orbitals} orbitals)`
                : "Fat bands cleared"
        );
    } catch (err) {
        setStatus(err.message, true);
    }
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
            if (dom.gapPredict) dom.gapPredict.disabled = true;
            if (dom.qeSlabPrepare) dom.qeSlabPrepare.disabled = true;
            setStatus("Load a CIF in the Crystal Suite first.");
            renderShells();
            return;
        }

        structures.forEach((s) => {
            const option = document.createElement("option");
            option.value = s.name;
            const base = Number(s.suggest_nbnd);
            const nbndNote = Number.isFinite(base) && base > 0 ? `, nbnd≈${base}` : "";
            option.textContent = `${s.name} (${s.formula}, ${s.n_sites} sites${nbndNote})`;
            dom.structures.appendChild(option);
        });

        // Keep first structure selected after rebuild so suggestions apply.
        if (!dom.structures.value || ![...dom.structures.options].some((o) => o.value === dom.structures.value)) {
            dom.structures.value = structures[0].name;
        }

        dom.calculate.disabled = false;
        if (dom.gapPredict) dom.gapPredict.disabled = false;
        if (dom.qeSlabPrepare) dom.qeSlabPrepare.disabled = false;
        // nbnd first — independent of TB shell rendering
        syncNbndSuggestion();
        syncSlabSuggestion();
        renderShells();
        setStatus(`${structures.length} structure(s) available`);
        refreshBzNote();
    } catch (err) {
        setStatus(err.message, true);
    }
}

function syncSlabSuggestion() {
    const structure = selected();
    if (!structure) return;
    if (dom.qeSlabMode) {
        dom.qeSlabMode.checked = !!structure.suggest_slab_qe;
    }
    if (dom.qeKz && structure.suggest_slab_qe) {
        dom.qeKz.value = 1;
    }
    if (dom.qeSlabStatus) {
        if (structure.suggest_slab_qe) {
            dom.qeSlabStatus.textContent =
                `Looks like slab/stack (c≈${(structure.lattice_c || 0).toFixed(1)} Å) — Slab QE suggested. Tab 3 stacks: Generate as-is. Bulk: Prepare slab first if needed.`;
        } else {
            dom.qeSlabStatus.textContent =
                "Bulk-like cell. Use Prepare slab (preset or custom hkl), or leave Slab QE off for 3D bulk.";
        }
    }
    if (dom.qeSlabStore && !dom.qeSlabStore.value) {
        dom.qeSlabStore.placeholder = `${structure.name}_slab`;
    }
}

function applySuggestedNbnd(structure) {
    const nbndEl = document.getElementById("qe-nbnd") || dom.qeNbnd;
    if (!structure || !nbndEl) return;
    const raw = structure.suggest_nbnd ?? structure.suggestNbnd;
    const base = Number(raw);
    if (!Number.isFinite(base) || base < 1) {
        setQeStatus(`No nbnd suggestion for ${structure.name || "structure"}`, true);
        return;
    }
    const socEl = document.getElementById("qe-soc") || dom.qeSoc;
    const soc = Boolean(socEl?.checked);
    const nbnd = Math.min(2000, Math.max(1, soc ? base * 2 : base));
    nbndEl.value = String(nbnd);
    const note = soc
        ? `Suggested nbnd=${nbnd} (${base}×2 SOC) for ${structure.name}`
        : `Suggested nbnd=${nbnd} for ${structure.name} (enable SOC → ${Math.min(2000, base * 2)})`;
    setQeStatus(note);
    const hint = document.getElementById("qe-nbnd-hint");
    if (hint) hint.textContent = note;
}

function syncNbndSuggestion() {
    applySuggestedNbnd(selected());
}

function setSlabStatus(message, isError = false) {
    if (!dom.qeSlabStatus) return;
    dom.qeSlabStatus.textContent = message;
    dom.qeSlabStatus.style.color = isError ? "#ff6b6b" : "";
}

async function prepareSlab() {
    const structure = selected();
    if (!structure) return;
    if (dom.qeSlabPrepare) dom.qeSlabPrepare.disabled = true;
    setSlabStatus(`Cleaving ${structure.name}\u2026`);
    try {
        const preset = dom.qeSlabPreset?.value || "thin_001";
        const storeAs = (dom.qeSlabStore?.value || "").trim() || `${structure.name}_slab`;
        const result = await TensorSpecAPI.dftPrepareSlab(structure.name, {
            preset,
            h: Number(dom.qeSlabH?.value) || 0,
            k: Number(dom.qeSlabK?.value) || 0,
            l: Number(dom.qeSlabL?.value) || 1,
            num_layers: Number(dom.qeSlabLayers?.value) || 1,
            vacuum: Number(dom.qeSlabVac?.value) || 15,
            store_as: storeAs,
        });
        setSlabStatus(
            `Stored ${result.stored_as} (${result.formula}, ${result.n_sites} sites) · hkl=${result.hkl.join("")} · ${result.num_layers}L · vac ${result.vacuum} Å`
        );
        await refreshStructures();
        if (dom.structures) {
            dom.structures.value = result.stored_as;
            renderShells();
            syncSlabSuggestion();
        }
        if (dom.qeSlabMode) dom.qeSlabMode.checked = true;
        if (dom.qeKz) dom.qeKz.value = 1;
        setStatus(`Slab ready: ${result.stored_as}`);
    } catch (err) {
        setSlabStatus(err.message, true);
        setStatus(err.message, true);
    } finally {
        if (dom.qeSlabPrepare) dom.qeSlabPrepare.disabled = false;
    }
}

async function refreshBzNote() {
    const structure = selected();
    if (!structure || !dom.pathNote) return;
    try {
        const ctx = await TensorSpecAPI.dftBzContext(structure.name);
        const mode = PATH_VALUES[dom.pathMode?.value] || "auto";
        if (mode === "unfold_hex") {
            dom.pathNote.textContent =
                "Unfold hex: supercell bands with Popescu–Zunger spectral weight on graphene-like Γ–K–M. Bright = monolayer character.";
        } else if (mode === "primitive_hex_ref") {
            dom.pathNote.textContent =
                "Primitive hex Γ–K–M–Γ in Å⁻¹, folded into this cell’s BZ. Educational — use Unfold hex for spectral weights.";
        } else if (ctx.likely_folded) {
            dom.pathNote.textContent = `${ctx.title}: ${ctx.message}`;
        } else {
            dom.pathNote.textContent = ctx.message || dom.pathNote.textContent;
        }
    } catch (_) {
        /* keep default hint */
    }
}

async function predictGap() {
    const structure = selected();
    if (!structure) return;
    if (dom.gapPredict) dom.gapPredict.disabled = true;
    if (dom.gapStatus) dom.gapStatus.textContent = "Predicting gap (may download MEGNet weights)…";
    try {
        const result = await TensorSpecAPI.dftGapPredict(
            structure.name,
            dom.gapFid?.value || "PBE"
        );
        if (dom.gapStatus) {
            dom.gapStatus.textContent =
                `Eg ≈ ${result.gap_eV.toFixed(3)} eV (${result.fidelity}) · ${result.formula} · scalar only`;
        }
        setStatus(`MEGNet gap ≈ ${result.gap_eV.toFixed(3)} eV`);
    } catch (err) {
        if (dom.gapStatus) dom.gapStatus.textContent = err.message;
        setStatus(err.message, true);
    } finally {
        if (dom.gapPredict) dom.gapPredict.disabled = false;
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
        use_wannier: Boolean(dom.w90Use?.checked),
        overlay_wannier: Boolean(dom.w90Overlay?.checked),
    };
}

function readQeParameters() {
    const slab = !!dom.qeSlabMode?.checked;
    syncNbndSuggestion();
    return {
        run_name: dom.qeRunName.value.trim() || "run_01",
        ecutwfc: Number(dom.qeCutoff.value) || 60,
        nbnd: Number(dom.qeNbnd.value) || 12,
        kx: Number(dom.qeKx.value) || 6,
        ky: Number(dom.qeKy.value) || 6,
        kz: slab ? 1 : (Number(dom.qeKz.value) || 6),
        use_soc: dom.qeSoc.checked,
        mlwf_mode: dom.qeWanMode.value === "mlwf",
        use_mpi: dom.qeMpi.checked,
        mpi_ranks: Number(dom.qeRanks.value) || 20,
        slab_mode: slab,
        functional: dom.qeXc?.value || "PBE",
        backend: dom.qeBackend?.value || "local",
    };
}

function readIsoenergyParameters() {
    const base = readParameters();
    return {
        energy: Number(dom.isoe?.value) || 0.0,
        kx_min: -2.0,
        kx_max: 2.0,
        ky_min: -2.0,
        ky_max: 2.0,
        resolution: 24,
        smear: 0.05,
        hoppings: base.hoppings,
        cutoffs: base.cutoffs,
        onsite_e: base.onsite_e,
        shift_s: base.shift_s,
        shift_p: base.shift_p,
        shift_d: base.shift_d,
        use_soc: base.use_soc,
        soc_strength: base.soc_strength,
        tb_mode: base.tb_mode,
        use_wannier: false,
    };
}

async function calculate() {
    const structure = selected();
    if (!structure) return;

    dom.calculate.disabled = true;
    setStatus(`Solving ${structure.formula}\u2026`);
    try {
        if (isIsoenergyMode()) {
            const result = await TensorSpecAPI.dftIsoenergy(
                structure.name,
                readIsoenergyParameters(),
            );
            lastCrystalName = structure.name;
            lastBandResult = null;
            populateFatOptions([]);
            renderIsoenergyHeatmap(result);
            dom.statBands.textContent =
                `${result.n_bands} bands, ${result.resolution}\u00d7${result.resolution} mesh`;
            dom.statTime.textContent = `solved in ${result.elapsed_seconds}s`;
            dom.badge.textContent = `isoenergy @ ${result.energy.toFixed(2)} eV`;
            setStatus(`Isoenergy map ${result.name}`);
            return;
        }

        const result = await TensorSpecAPI.dftBands(structure.name, readParameters());
        lastCrystalName = structure.name;
        lastBandResult = { ...result, fat_weights: null, fat_target: "none" };
        populateFatOptions(result.orbital_labels || []);

        const view = ensurePlot();
        view.setResult(lastBandResult);
        view.setRange(Number(dom.eMin.value) || -6, Number(dom.eMax.value) || 6);

        dom.statBands.textContent = `${result.n_bands} bands, ${result.n_kpoints} k-points`;
        dom.statTime.textContent = `solved in ${result.elapsed_seconds}s`;
        dom.badge.textContent = result.node_labels.join(" \u2192 ") || "computed";
        if (dom.pathNote && result.path_note) {
            const title = result.path_title ? `${result.path_title}. ` : "";
            dom.pathNote.textContent = `${title}${result.path_note}`;
        }
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
        applyQueueEnable(lastSolversInfo);
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

async function loadWannierHr() {
    const structure = selected();
    if (!structure) {
        setStatus("Select a crystal first.", true);
        return;
    }
    if (!dom.w90File) return;
    dom.w90File.value = "";
    dom.w90File.click();
}

async function onWannierFileChosen(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const structure = selected();
    if (!structure) {
        setStatus("Select a crystal first.", true);
        return;
    }
    if (dom.w90Load) dom.w90Load.disabled = true;
    if (dom.w90Status) dom.w90Status.textContent = `Uploading ${file.name}…`;
    try {
        const result = await TensorSpecAPI.dftUploadWannier(structure.name, file);
        if (dom.w90Use) {
            dom.w90Use.disabled = false;
            dom.w90Use.checked = true;
        }
        if (dom.w90Overlay) {
            dom.w90Overlay.disabled = false;
        }
        if (dom.w90Status) {
            const scf = result.scf_out_saved ? " (+ scf.out)" : "";
            dom.w90Status.textContent =
                `Loaded ${file.name} (${result.bytes} bytes)${scf}. Check “Use uploaded W90” and/or Overlay, then Calculate.`;
        }
        setStatus(`Wannier HR stored for ${structure.name}`);
    } catch (err) {
        if (dom.w90Status) dom.w90Status.textContent = err.message;
        setStatus(err.message, true);
    } finally {
        if (dom.w90Load) dom.w90Load.disabled = false;
    }
}

function applyQueueEnable(info) {
    const einstein = dom.qeBackend?.value === "einstein_ssh";
    if (!info) {
        dom.qeQueue.disabled = !einstein;
        if (einstein) {
            setQeStatus("Einstein (SSH) queue enabled (solvers status pending)");
        }
        return;
    }
    if (info.available) {
        setQeStatus(
            `Solvers ready — max ${maxMpiRanks} MPI ranks`
            + (info.mpirun ? "" : " (mpirun not found; runs will be serial)")
        );
        dom.qeQueue.disabled = false;
    } else if (einstein) {
        setQeStatus(
            `Local solvers unavailable (${info.detail || "check server config"}); Einstein (SSH) queue still enabled`
        );
        dom.qeQueue.disabled = false;
    } else {
        setQeStatus(`Solvers unavailable: ${info.detail || "check server config"}`, true);
        dom.qeQueue.disabled = true;
    }
}

async function refreshSolvers() {
    try {
        const info = await TensorSpecAPI.dftSolvers();
        lastSolversInfo = info;
        maxMpiRanks = info.max_mpi_ranks || 20;
        if (dom.qeRanks) {
            dom.qeRanks.max = maxMpiRanks;
            const cur = Number(dom.qeRanks.value);
            if (!Number.isFinite(cur) || cur < 1 || cur > maxMpiRanks) {
                dom.qeRanks.value = String(maxMpiRanks);
            }
        }
        applyQueueEnable(info);
    } catch (err) {
        setQeStatus(err.message, true);
        dom.qeQueue.disabled = dom.qeBackend?.value !== "einstein_ssh";
    }
}

dom.refresh.addEventListener("click", refreshStructures);
dom.structures.addEventListener("change", () => {
    renderShells();
    refreshBzNote();
    syncSlabSuggestion();
    syncNbndSuggestion();
});
if (dom.qeSoc) {
    dom.qeSoc.addEventListener("change", () => syncNbndSuggestion());
}
dom.calculate.addEventListener("click", calculate);
if (dom.gapPredict) dom.gapPredict.addEventListener("click", predictGap);
if (dom.fat) dom.fat.addEventListener("change", applyFatTarget);
if (dom.w90Load) dom.w90Load.addEventListener("click", loadWannierHr);
if (dom.w90File) dom.w90File.addEventListener("change", onWannierFileChosen);
if (dom.w90Use) {
    dom.w90Use.addEventListener("change", () => {
        if (dom.w90Status) {
            dom.w90Status.textContent = dom.w90Use.checked
                ? "Status: Calculate will use uploaded Wannier90 Hamiltonian."
                : "Status: Using Manual Slater-Koster parameters.";
        }
    });
}
if (dom.qeSlabPrepare) dom.qeSlabPrepare.addEventListener("click", prepareSlab);
if (dom.qeSlabMode) {
    dom.qeSlabMode.addEventListener("change", () => {
        if (dom.qeSlabMode.checked && dom.qeKz) dom.qeKz.value = 1;
    });
}
if (dom.qeBackend) {
    dom.qeBackend.addEventListener("change", () => {
        applyQueueEnable(lastSolversInfo);
    });
}
dom.qeGenerate.addEventListener("click", generateInputs);
dom.qeBundle.addEventListener("click", downloadBundle);
dom.qeQueue.addEventListener("click", queueRun);
dom.qeCancel.addEventListener("click", cancelRun);

if (dom.soc) {
    dom.soc.addEventListener("change", () => {
        if (dom.socStrength) dom.socStrength.disabled = !dom.soc.checked;
    });
}

[dom.eMin, dom.eMax].forEach((input) => {
    if (!input) return;
    input.addEventListener("change", () => {
        if (plot) plot.setRange(Number(dom.eMin.value) || -6, Number(dom.eMax.value) || 6);
    });
});

function syncPathFields() {
    syncKgridMode();
    if (isIsoenergyMode()) return;
    if (!dom.pathMode) return;
    const isCustom = PATH_VALUES[dom.pathMode.value] === "custom";
    if (dom.coords) dom.coords.disabled = !isCustom;
    if (dom.labels) dom.labels.disabled = !isCustom;
    refreshBzNote();
}
if (dom.pathMode) dom.pathMode.addEventListener("change", syncPathFields);
if (dom.kgrid) dom.kgrid.addEventListener("change", syncPathFields);
try {
    syncPathFields();
} catch (err) {
    console.error("[dft] syncPathFields failed", err);
}

const api = globalThis.TensorSpecAPI;
if (!api) {
    setStatus("TensorSpecAPI missing — reload page (api.js failed to bind).", true);
} else {
    refreshStructures().catch((err) => {
        console.error("[dft] refreshStructures failed", err);
        setStatus(String(err.message || err), true);
    });
    refreshSolvers().catch((err) => {
        console.error("[dft] refreshSolvers failed", err);
        setQeStatus(String(err.message || err), true);
    });
}

window.addEventListener("pageshow", () => {
    try {
        syncNbndSuggestion();
    } catch (err) {
        console.error("[dft] pageshow nbnd sync failed", err);
    }
});
