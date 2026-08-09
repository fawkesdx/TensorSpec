/* Crystal Suite controller: Tabs 1–4.
 *
 * Moves values between the DOM and the server. CDW, stack, and BZ math all
 * live in core.crystallography; this file only orchestrates requests.
 */

import { CrystalViewer, elementColor, bondColor } from "/static/js/viewers/viewer_3d.js";

const el = (id) => document.getElementById(id);

const dom = {
    file: el("cr-file"),
    load: el("cr-load"),
    status: el("cr-status"),
    spacegroup: el("cr-spacegroup"),
    badge: el("cr-session-badge"),
    legend: el("cr-legend"),
    viewport: el("cr-viewport"),
    placeholder: el("cr-placeholder"),
    nx: el("cr-nx"),
    ny: el("cr-ny"),
    nz: el("cr-nz"),
    threshold: el("cr-thresh"),
    radius: el("cr-radius"),
    bondThick: el("cr-bondthick"),
    swatches: el("cr-swatches"),
    conn: el("cr-conn"),
    pbr: el("cr-pbr"),
    axes: el("cr-axes"),
    showCell: el("cr-show-cell"),
    projection: el("cr-projection"),
    viewA: el("cr-view-a"),
    viewB: el("cr-view-b"),
    viewC: el("cr-view-c"),
    view111: el("cr-view-111"),
    az: el("cr-az"),
    elv: el("cr-el"),
    basisConv: el("cr-basis-conv"),
    basisPrim: el("cr-basis-prim"),
    statAtoms: el("cr-stat-atoms"),
    statBonds: el("cr-stat-bonds"),
    cutH: el("cr-h"),
    cutK: el("cr-k"),
    cutL: el("cr-l"),
    align: el("cr-align"),
    cut: el("cr-cut"),
    cutColor: el("cr-cut-color"),
    depth: el("cr-depth"),
    expAtoms: el("cr-exp-atoms"),
    expCell: el("cr-exp-cell"),
    expBz: el("cr-exp-bz"),
    export3ds: el("cr-export-3ds"),
    exportBlender: el("cr-export-blender"),
    crRender: el("cr-render"),
    crHires: el("cr-hires"),

    cdwEnable: el("cdw-enable"),
    cdwTarget: el("cdw-target"),
    qa: el("q-a"),
    qb: el("q-b"),
    qc: el("q-c"),
    ax: el("a-x"),
    ay: el("a-y"),
    cdwAz: el("a-z"),
    cdwPhase: el("cdw-phase"),

    stTemplate: el("st-template"),
    stAdd: el("st-add-template"),
    stLayers: el("st-layers"),
    stRender: el("st-render"),
    stMoire: el("st-moire"),
    stStatus: el("st-status"),
    stStore: el("st-store"),
    stMlip: el("st-mlip"),
    stFmax: el("st-fmax"),
    stSteps: el("st-steps"),
    stRelaxCell: el("st-relax-cell"),
    stRelax: el("st-relax"),
    stGapFid: el("st-gap-fid"),
    stGap: el("st-gap"),
    stCif: el("st-cif"),
    stPush: el("st-push"),
    crExportCif: el("cr-export-cif"),
    crPush: el("cr-push"),
    eraser: el("cr-eraser"),
    eraserReset: el("cr-eraser-reset"),

    exSource: el("ex-source"),
    exRefresh: el("ex-refresh"),
    exMode: el("ex-mode"),
    exMiller: el("ex-miller-opts"),
    exLayers: el("ex-layers"),
    exVacuum: el("ex-vacuum"),
    exH: el("ex-h"),
    exK: el("ex-k"),
    exL: el("ex-l"),
    exRun: el("ex-run"),

    bzOverlay: el("bz-overlay"),
    bzScale: el("bz-scale"),
    bzStyle: el("bz-style"),
    bzSurface: el("bz-surface"),
    bzH: el("bz-h"),
    bzK: el("bz-k"),
    bzL: el("bz-l"),
    bzRender: el("bz-render"),
    bzClear: el("bz-clear"),
};

let viewer = null;
let activeCrystal = null;
const stackLayers = [];

function setStatus(message, isError = false) {
    dom.status.textContent = message;
    dom.status.style.color = isError ? "#ff6b6b" : "";
}

function setStackStatus(message, isError = false) {
    dom.stStatus.textContent = message;
    dom.stStatus.style.color = isError ? "#ff6b6b" : "";
}

function ensureViewer() {
    if (viewer) return viewer;
    if (dom.placeholder) dom.placeholder.remove();
    viewer = new CrystalViewer(dom.viewport);
    return viewer;
}

function fillTargets(elements) {
    const previous = dom.cdwTarget.value;
    dom.cdwTarget.innerHTML = '<option>All Elements</option>';
    elements.forEach((symbol) => {
        const option = document.createElement("option");
        option.textContent = symbol;
        option.value = symbol;
        dom.cdwTarget.appendChild(option);
    });
    if ([...dom.cdwTarget.options].some((o) => o.value === previous)) {
        dom.cdwTarget.value = previous;
    }
}

function shouldRevertPrimitiveBasis(err) {
    if (!dom.basisPrim?.checked) return false;
    if (err?.status === 422) return true;
    const msg = String(err?.message || "").toLowerCase();
    return msg.includes("primitive") || /\b422\b/.test(msg);
}

function rebuildSwatches(elements) {
    if (!dom.swatches) return;
    dom.swatches.innerHTML = "";
    elements.forEach((symbol) => {
        const row = document.createElement("div");
        row.className = "swatch-row";
        const label = document.createElement("span");
        label.textContent = `${symbol}:`;
        const input = document.createElement("input");
        input.type = "color";
        input.value = elementColor(symbol);
        input.addEventListener("input", () => {
            const view = ensureViewer();
            view.setElementColor(symbol, input.value);
            updateLegend(elements);
        });
        row.append(label, input);
        dom.swatches.appendChild(row);
    });
    const bondRow = document.createElement("div");
    bondRow.className = "swatch-row";
    const bondLabel = document.createElement("span");
    bondLabel.textContent = "Bonds:";
    const bondInput = document.createElement("input");
    bondInput.type = "color";
    bondInput.value = bondColor();
    bondInput.addEventListener("input", () => ensureViewer().setBondColor(bondInput.value));
    bondRow.append(bondLabel, bondInput);
    dom.swatches.appendChild(bondRow);
}

function updateLegend(elements) {
    dom.legend.textContent = elements.join(", ");
    if (elements[0]) dom.legend.style.color = elementColor(elements[0]);
}

function applyCutPlane(view) {
    if (!view) return;
    const h = Number(dom.cutH?.value) || 0;
    const k = Number(dom.cutK?.value) || 0;
    const l = Number(dom.cutL?.value) || 0;
    if (dom.cut?.checked && h === 0 && k === 0 && l === 0) {
        setStatus("Cut plane needs a non-zero [h k l].", true);
        view.setCutPlane({ visible: false });
        return;
    }
    if (!activeCrystal && dom.cut?.checked) {
        setStatus("Load a structure first.", true);
        return;
    }
    view.setCutPlane({
        h, k, l,
        depthFrac: (Number(dom.depth?.value) || 0) / 100,
        color: dom.cutColor?.value || "#00ffff",
        visible: Boolean(dom.cut?.checked),
    });
}

function applyViewerChrome(view) {
    view.atomScale = Number(dom.radius.value) || 0.5;
    view.setBondRadius(Number(dom.bondThick?.value) || 0.1);
    view.setShowAxes(Boolean(dom.axes?.checked));
    view.setPbrShiny(Boolean(dom.pbr?.checked));
    applyCutPlane(view);
}

function omittedAtomIndices() {
    return viewer ? viewer.getOmittedAtomIndices() : [];
}

function drawExportKnobs() {
    const geo = geometryRequest();
    return {
        nx: geo.nx,
        ny: geo.ny,
        nz: geo.nz,
        basis: geo.basis,
        omit_atom_indices: omittedAtomIndices(),
    };
}

function geometryRequest() {
    return {
        nx: Number(dom.nx.value) || 1,
        ny: Number(dom.ny.value) || 1,
        nz: Number(dom.nz.value) || 1,
        bond_threshold: Number(dom.threshold.value) || 1.15,
        basis: dom.basisPrim?.checked ? "primitive" : "conventional",
        show_bonds: dom.conn?.value !== "none",
        cdw_enabled: Boolean(dom.cdwEnable.checked),
        cdw_target: dom.cdwTarget.value || "All Elements",
        cdw_qx: Number(dom.qa.value) || 0,
        cdw_qy: Number(dom.qb.value) || 0,
        cdw_qz: Number(dom.qc.value) || 0,
        cdw_ax: Number(dom.ax.value) || 0,
        cdw_ay: Number(dom.ay.value) || 0,
        cdw_az: Number(dom.cdwAz.value) || 0,
        cdw_phase: Number(dom.cdwPhase.value) || 0,
    };
}

async function refreshGeometry({ frame = true } = {}) {
    if (!activeCrystal) return;

    setStatus(`Building geometry for ${activeCrystal}\u2026`);
    try {
        const geometry = await TensorSpecAPI.crystalGeometry(activeCrystal, geometryRequest());
        const view = ensureViewer();
        view.clearErasedAtoms();
        applyViewerChrome(view);
        view.render(geometry, {
            frame,
            showBonds: dom.conn?.value !== "none",
            showCell: Boolean(dom.showCell?.checked),
        });
        rebuildSwatches(geometry.elements);
        updateLegend(geometry.elements);

        dom.statAtoms.textContent = `${geometry.n_atoms} atoms`;
        dom.statBonds.textContent = `${geometry.bonds.length} bonds`;
        fillTargets(geometry.elements);
        setStatus(`${activeCrystal} rendered`);
    } catch (err) {
        setStatus(err.message, true);
        if (shouldRevertPrimitiveBasis(err)) {
            dom.basisConv.checked = true;
            dom.basisPrim.checked = false;
        }
    }
}

async function onFileChosen(event) {
    const file = event.target.files[0];
    if (!file) return;

    setStatus(`Parsing ${file.name}\u2026`);
    try {
        const summary = await TensorSpecAPI.loadCif(file);
        activeCrystal = summary.name;
        dom.spacegroup.textContent =
            `Space Group: ${summary.spacegroup} \u2014 ${summary.formula}, ${summary.n_sites} sites`;
        await refreshGeometry({ frame: true });
        await refreshExfoliateSources();
        if (dom.exSource) dom.exSource.value = summary.name;
    } catch (err) {
        setStatus(err.message, true);
    } finally {
        event.target.value = "";
    }
}

function renderLayerRows() {
    dom.stLayers.innerHTML = "";
    if (!stackLayers.length) {
        dom.stLayers.innerHTML = '<p class="hint">No layers yet. Add a template above.</p>';
        return;
    }

    stackLayers.forEach((layer, index) => {
        const row = document.createElement("div");
        row.className = "layer-row";
        row.innerHTML = `
            <span class="layer-row__name">${layer.label}</span>
            <label>SC:</label>
            <input class="field field--num" data-k="sc_x" type="number" min="1" max="50" value="${layer.sc_x}">
            <input class="field field--num" data-k="sc_y" type="number" min="1" max="50" value="${layer.sc_y}">
            <label>z (&#197;):</label>
            <input class="field field--num" data-k="z_shift" type="number" min="-100" max="100" step="0.5" value="${layer.z_shift}">
            <label>&#952; (&#176;):</label>
            <input class="field field--num" data-k="twist" type="number" min="-360" max="360" step="1" value="${layer.twist}">
            <button type="button" class="btn" data-up title="Move up">&#9650;</button>
            <button type="button" class="btn" data-down title="Move down">&#9660;</button>
            <button type="button" class="btn" data-remove>X</button>`;

        row.querySelectorAll("input").forEach((input) => {
            input.addEventListener("change", () => {
                layer[input.dataset.k] = Number(input.value);
            });
        });
        row.querySelector("[data-up]").addEventListener("click", () => {
            if (index === 0) return;
            [stackLayers[index - 1], stackLayers[index]] = [stackLayers[index], stackLayers[index - 1]];
            renderLayerRows();
        });
        row.querySelector("[data-down]").addEventListener("click", () => {
            if (index >= stackLayers.length - 1) return;
            [stackLayers[index + 1], stackLayers[index]] = [stackLayers[index], stackLayers[index + 1]];
            renderLayerRows();
        });
        row.querySelector("[data-remove]").addEventListener("click", () => {
            stackLayers.splice(index, 1);
            renderLayerRows();
        });

        dom.stLayers.appendChild(row);
    });
}

async function addTemplate() {
    const templateName = dom.stTemplate.value;
    setStackStatus(`Adding ${templateName}\u2026`);
    try {
        const summary = await TensorSpecAPI.crystalAddTemplate({ template_name: templateName });
        const zShift = stackLayers.length ? Math.max(...stackLayers.map((l) => l.z_shift)) + 3.4 : 0;
        stackLayers.push({
            name: summary.name,
            label: `${summary.name} (${summary.formula})`,
            sc_x: 1,
            sc_y: 1,
            z_shift: Number(zShift.toFixed(1)),
            twist: 0,
        });
        renderLayerRows();
        setStackStatus(`Added ${summary.name}`);
    } catch (err) {
        setStackStatus(err.message, true);
    }
}

async function renderStack() {
    if (!stackLayers.length) {
        setStackStatus("Add at least one layer first.", true);
        return;
    }
    setStackStatus("Building stack\u2026");
    try {
        const storeAs = (dom.stStore?.value || "heterostructure").trim() || "heterostructure";
        const geometry = await TensorSpecAPI.crystalStack({
            layers: stackLayers.map(({ name, sc_x, sc_y, z_shift, twist }) => ({
                name, sc_x, sc_y, z_shift, twist,
            })),
            store_as: storeAs,
            show_bonds: dom.conn?.value !== "none",
        });
        activeCrystal = geometry.name;
        const view = ensureViewer();
        view.clearBrillouinZone();
        view.clearErasedAtoms();
        applyViewerChrome(view);
        view.render(geometry, {
            frame: true,
            showBonds: dom.conn?.value !== "none",
            showCell: Boolean(dom.showCell?.checked),
        });
        fillTargets(geometry.elements);
        dom.statAtoms.textContent = `${geometry.n_atoms} atoms`;
        dom.statBonds.textContent = `${geometry.bonds.length} bonds`;
        rebuildSwatches(geometry.elements);
        updateLegend(geometry.elements);
        setStackStatus(`Rendered ${geometry.name} (${geometry.n_atoms} atoms)`);
        setStatus(`${geometry.name} active`);
    } catch (err) {
        setStackStatus(err.message, true);
    }
}

async function refreshExfoliateSources() {
    if (!dom.exSource) return;
    try {
        const listing = await TensorSpecAPI.listItems();
        const crystals = listing.items.filter((item) => /crystal/i.test(item.type));
        const previous = dom.exSource.value || activeCrystal || "";
        dom.exSource.innerHTML = "";
        if (!crystals.length) {
            dom.exSource.innerHTML = '<option value="">Load a CIF in Tab 1 first</option>';
            return;
        }
        crystals.forEach((item) => {
            const option = document.createElement("option");
            option.value = item.name;
            option.textContent = `${item.name} — ${item.dims}`;
            dom.exSource.appendChild(option);
        });
        if ([...dom.exSource.options].some((o) => o.value === previous)) {
            dom.exSource.value = previous;
        }
    } catch (err) {
        setStackStatus(err.message, true);
    }
}

function syncExfoliateMode() {
    const miller = dom.exMode.value === "miller";
    dom.exMiller.hidden = !miller;
}

async function runExfoliate() {
    const source = dom.exSource.value;
    if (!source) {
        setStackStatus("Load a bulk CIF in Tab 1, then pick it as the source.", true);
        return;
    }
    const mode = dom.exMode.value;
    setStackStatus(`Exfoliating ${source} (${mode})\u2026`);
    try {
        const result = await TensorSpecAPI.crystalExfoliate({
            source_name: source,
            mode,
            h: Number(dom.exH.value) || 0,
            k: Number(dom.exK.value) || 0,
            l: Number(dom.exL.value) || 1,
            num_layers: Number(dom.exLayers.value) || 1,
            vacuum: Number(dom.exVacuum.value) || 15,
        });
        const summary = result.summary;
        const zShift = stackLayers.length
            ? Math.max(...stackLayers.map((layer) => layer.z_shift)) + 3.4
            : 0;
        stackLayers.push({
            name: summary.name,
            label: `${summary.name} (${summary.formula})`,
            sc_x: 1,
            sc_y: 1,
            z_shift: Number(zShift.toFixed(1)),
            twist: 0,
        });
        renderLayerRows();

        activeCrystal = summary.name;
        await refreshGeometry({ frame: true });
        await refreshExfoliateSources();
        if (dom.exSource) dom.exSource.value = summary.name;

        const gapNote = result.gap_angstrom != null
            ? `; vdW gap ${result.gap_angstrom.toFixed(2)} Å`
            : "";
        const hklNote = result.hkl
            ? `; [${result.hkl.join(" ")}], ${result.num_layers} layer(s)`
            : "";
        setStackStatus(`Extracted ${summary.name} (${summary.n_sites} sites${gapNote}${hklNote})`);
        setStatus(`${summary.name} active`);
    } catch (err) {
        setStackStatus(err.message, true);
    }
}

async function calculateMoire() {
    if (stackLayers.length !== 2) {
        setStackStatus("Moiré needs exactly two layers.", true);
        return;
    }
    const [a, b] = stackLayers;
    try {
        const result = await TensorSpecAPI.crystalMoire({
            layer1: a.name,
            layer2: b.name,
            twist1: a.twist,
            twist2: b.twist,
            z_min: Math.min(a.z_shift, b.z_shift) - 2,
            z_max: Math.max(a.z_shift, b.z_shift) + 2,
        });
        if (result.envelope) {
            ensureViewer().setMoireEnvelope(result.envelope, { frame: false });
        }
        const period = result.periodicity != null ? `${result.periodicity.toFixed(2)} Å` : "—";
        const extra = result.n_cells != null ? `, ~${result.n_cells}×${result.n_cells} cells` : "";
        setStackStatus(`Moiré ${result.status}: period ${period}${extra}`);
    } catch (err) {
        setStackStatus(err.message, true);
    }
}

async function renderBZ() {
    if (!activeCrystal) {
        setStatus("Load a crystal first.", true);
        return;
    }
    setStatus(`Building BZ for ${activeCrystal}\u2026`);
    try {
        const bz = await TensorSpecAPI.crystalBZ(activeCrystal, {
            scale: Number(dom.bzScale.value) || 1,
            style: dom.bzStyle.value || "solid",
            surface: Boolean(dom.bzSurface.checked),
            h: Number(dom.bzH.value) || 0,
            k: Number(dom.bzK.value) || 0,
            l: Number(dom.bzL.value) || 1,
        });

        const view = ensureViewer();
        if (!dom.bzOverlay.checked) {
            view.geometry = null;
            view.clear();
            view.setBrillouinZone(bz, { frame: true });
        } else {
            if (!view.geometry) await refreshGeometry({ frame: false });
            view.setBrillouinZone(bz, { frame: true });
        }
        setStatus(`BZ rendered (${bz.style}, scale ${bz.scale})`);
    } catch (err) {
        setStatus(err.message, true);
    }
}

dom.load.addEventListener("click", () => dom.file.click());
dom.file.addEventListener("change", onFileChosen);

[dom.nx, dom.ny, dom.nz, dom.threshold].forEach((node) =>
    node?.addEventListener("change", () => refreshGeometry({ frame: false }))
);
dom.basisConv?.addEventListener("change", () => refreshGeometry({ frame: true }));
dom.basisPrim?.addEventListener("change", () => refreshGeometry({ frame: true }));
dom.conn?.addEventListener("change", () => refreshGeometry({ frame: false }));
dom.bondThick?.addEventListener("change", () => {
    ensureViewer().setBondRadius(Number(dom.bondThick.value) || 0.1);
});
dom.radius?.addEventListener("change", () => {
    ensureViewer().setAtomScale(Number(dom.radius.value) || 0.5);
});
dom.pbr?.addEventListener("change", () => ensureViewer().setPbrShiny(dom.pbr.checked));
dom.axes?.addEventListener("change", () => ensureViewer().setShowAxes(dom.axes.checked));
dom.showCell?.addEventListener("change", () => refreshGeometry({ frame: false }));
dom.projection?.addEventListener("change", () => ensureViewer().setProjection(dom.projection.value));
dom.viewA?.addEventListener("click", () => ensureViewer().lookAlong("+a"));
dom.viewB?.addEventListener("click", () => ensureViewer().lookAlong("+b"));
dom.viewC?.addEventListener("click", () => ensureViewer().lookAlong("+c"));
dom.view111?.addEventListener("click", () => ensureViewer().lookAlong("111"));
dom.align?.addEventListener("click", () => {
    const h = Number(dom.cutH?.value) || 0;
    const k = Number(dom.cutK?.value) || 0;
    const l = Number(dom.cutL?.value) || 0;
    if (h === 0 && k === 0 && l === 0) {
        setStatus("Align needs non-zero [h k l].", true);
        return;
    }
    ensureViewer().lookAlongMiller(h, k, l);
    setStatus(`Aligned to [${h} ${k} ${l}]`);
});
const syncAzEl = () => ensureViewer().setAzEl(Number(dom.az.value) || 0, Number(dom.elv.value) || 0);
dom.az?.addEventListener("change", syncAzEl);
dom.elv?.addEventListener("change", syncAzEl);

const syncCut = () => applyCutPlane(ensureViewer());
[dom.cutH, dom.cutK, dom.cutL, dom.cutColor].forEach((n) => n?.addEventListener("change", syncCut));
dom.cut?.addEventListener("change", syncCut);
dom.depth?.addEventListener("input", syncCut);

dom.eraser?.addEventListener("change", () => {
    ensureViewer().setEraserEnabled(dom.eraser.checked);
});
dom.eraserReset?.addEventListener("click", () => {
    ensureViewer().clearErasedAtoms();
    setStatus("Erase reset");
});

[
    dom.cdwEnable, dom.cdwTarget, dom.qa, dom.qb, dom.qc,
    dom.ax, dom.ay, dom.cdwAz, dom.cdwPhase,
].forEach((input) =>
    input.addEventListener("change", () => refreshGeometry({ frame: false }))
);

dom.stAdd.addEventListener("click", addTemplate);
dom.stRender.addEventListener("click", renderStack);
dom.stMoire.addEventListener("click", calculateMoire);
dom.exRefresh.addEventListener("click", refreshExfoliateSources);
dom.exMode.addEventListener("change", syncExfoliateMode);
dom.exRun.addEventListener("click", runExfoliate);
dom.bzRender.addEventListener("click", renderBZ);
dom.bzClear.addEventListener("click", () => {
    if (viewer) viewer.clearBrillouinZone();
    setStatus("BZ cleared");
});

async function downloadActiveCif() {
    if (!activeCrystal) {
        setStatus("Load or stack a structure first.", true);
        return;
    }
    setStatus(`Exporting CIF for ${activeCrystal}\u2026`);
    try {
        const blob = await TensorSpecAPI.crystalCifDownload(activeCrystal, drawExportKnobs());
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${activeCrystal}.cif`;
        link.click();
        URL.revokeObjectURL(url);
        const omit = omittedAtomIndices();
        const note = omit.length ? ` (${omit.length} atom(s) omitted)` : "";
        setStatus(`Exported ${activeCrystal}.cif${note}`);
        if (omit.length) {
            ensureViewer().clearErasedAtoms();
            await refreshGeometry({ frame: false });
        }
    } catch (err) {
        setStatus(err.message, true);
    }
}

function sceneExportPayload() {
    const geo = geometryRequest();
    return {
        nx: geo.nx,
        ny: geo.ny,
        nz: geo.nz,
        bond_threshold: geo.bond_threshold,
        basis: geo.basis,
        show_bonds: geo.show_bonds,
        include_atoms: Boolean(dom.expAtoms?.checked),
        include_cell: Boolean(dom.expCell?.checked),
        include_bz: Boolean(dom.expBz?.checked),
        bz_scale: Number(dom.bzScale?.value) || 1,
        bz_style: dom.bzStyle?.value || "solid",
        bz_h: Number(dom.bzH?.value) || 0,
        bz_k: Number(dom.bzK?.value) || 0,
        bz_l: Number(dom.bzL?.value) || 1,
        omit_atom_indices: omittedAtomIndices(),
    };
}

async function exportScene(fmt) {
    if (!activeCrystal) {
        setStatus("Load or stack a structure first.", true);
        return;
    }
    const payload = sceneExportPayload();
    if (!payload.include_atoms && !payload.include_cell && !payload.include_bz) {
        setStatus("Select at least one of Atoms/Bonds, Unit Cell, or Brillouin Zone.", true);
        return;
    }
    setStatus(`Exporting ${fmt} scene for ${activeCrystal}\u2026`);
    try {
        const blob = await TensorSpecAPI.crystalExportScene(activeCrystal, fmt, payload);
        const ext = fmt === "3dsmax" ? "ms" : "py";
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${activeCrystal}_scene.${ext}`;
        link.click();
        URL.revokeObjectURL(url);
        setStatus(`Exported ${activeCrystal}_scene.${ext}`);
    } catch (err) {
        setStatus(err.message, true);
    }
}

async function pushActiveCrystal(preferredName) {
    if (!activeCrystal) {
        setStatus("Nothing to push — load or render a stack first.", true);
        return;
    }
    const storeAs = (preferredName || dom.stStore?.value || activeCrystal).trim();
    if (!storeAs) {
        setStatus("Enter a workspace name.", true);
        return;
    }
    setStackStatus(`Pushing as ${storeAs}\u2026`);
    try {
        const summary = await TensorSpecAPI.crystalPush(activeCrystal, {
            store_as: storeAs,
            ...drawExportKnobs(),
        });
        activeCrystal = summary.name;
        if (dom.stStore) dom.stStore.value = summary.name;
        ensureViewer().clearErasedAtoms();
        await refreshGeometry({ frame: false });
        setStackStatus(`Pushed ${summary.name} (${summary.formula}, ${summary.n_sites} sites) — available in DFT Suite`);
        setStatus(`${summary.name} in workspace`);
    } catch (err) {
        setStackStatus(err.message, true);
        setStatus(err.message, true);
    }
}

async function relaxActiveStack() {
    if (!activeCrystal) {
        setStackStatus("Render a stack first, then relax.", true);
        return;
    }
    const storeAs = `${(dom.stStore?.value || activeCrystal).trim() || activeCrystal}_relaxed`;
    setStackStatus(`Relaxing ${activeCrystal} with ${dom.stMlip.value}\u2026 (first run may download weights)`);
    if (dom.stRelax) dom.stRelax.disabled = true;
    try {
        const result = await TensorSpecAPI.crystalRelax(activeCrystal, {
            model: dom.stMlip.value,
            fmax: Number(dom.stFmax.value) || 0.1,
            steps: Number(dom.stSteps.value) || 200,
            relax_cell: !!dom.stRelaxCell?.checked,
            store_as: storeAs,
            show_bonds: dom.conn?.value !== "none",
        });
        activeCrystal = result.stored_as;
        if (dom.stStore) dom.stStore.value = result.stored_as;
        const view = ensureViewer();
        view.clearBrillouinZone();
        view.clearErasedAtoms();
        applyViewerChrome(view);
        view.render(result.geometry, {
            frame: true,
            showBonds: dom.conn?.value !== "none",
            showCell: Boolean(dom.showCell?.checked),
        });
        fillTargets(result.geometry.elements);
        dom.statAtoms.textContent = `${result.geometry.n_atoms} atoms`;
        dom.statBonds.textContent = `${result.geometry.bonds.length} bonds`;
        rebuildSwatches(result.geometry.elements);
        updateLegend(result.geometry.elements);
        const e = result.final_energy_eV != null ? ` · E=${result.final_energy_eV.toFixed(3)} eV` : "";
        setStackStatus(
            `Relaxed → ${result.stored_as} (${result.model}${e}). Ready for DFT Suite / CIF.`
        );
        setStatus(`${result.stored_as} active`);
    } catch (err) {
        setStackStatus(err.message, true);
    } finally {
        if (dom.stRelax) dom.stRelax.disabled = false;
    }
}

async function predictStackGap() {
    if (!activeCrystal) {
        setStackStatus("Render or load a stack first.", true);
        return;
    }
    if (dom.stGap) dom.stGap.disabled = true;
    setStackStatus(`MEGNet gap for ${activeCrystal}\u2026`);
    try {
        const result = await TensorSpecAPI.crystalGapPredict(
            activeCrystal,
            dom.stGapFid?.value || "PBE"
        );
        setStackStatus(
            `Eg ≈ ${result.gap_eV.toFixed(3)} eV (${result.fidelity}) · ${result.formula} · scalar only — use DFT Suite for E(k)`
        );
    } catch (err) {
        setStackStatus(err.message, true);
    } finally {
        if (dom.stGap) dom.stGap.disabled = false;
    }
}

if (dom.stRelax) dom.stRelax.addEventListener("click", relaxActiveStack);
if (dom.stGap) dom.stGap.addEventListener("click", predictStackGap);
if (dom.stCif) dom.stCif.addEventListener("click", downloadActiveCif);
if (dom.stPush) dom.stPush.addEventListener("click", () => pushActiveCrystal());
if (dom.crExportCif) dom.crExportCif.addEventListener("click", downloadActiveCif);
if (dom.export3ds) dom.export3ds.addEventListener("click", () => exportScene("3dsmax"));
if (dom.exportBlender) dom.exportBlender.addEventListener("click", () => exportScene("blender"));
dom.crRender?.addEventListener("click", () => refreshGeometry({ frame: true }));
dom.crHires?.addEventListener("click", () => {
    if (!activeCrystal) { setStatus("Load a crystal first.", true); return; }
    const url = ensureViewer().capturePNG(2);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeCrystal || "structure"}.png`;
    a.click();
    setStatus("Saved high-res PNG");
});
if (dom.crPush) {
    dom.crPush.addEventListener("click", () => {
        const name = window.prompt("Workspace name for this structure:", activeCrystal || "structure");
        if (name) pushActiveCrystal(name);
    });
}

// Prefill MLIP availability hint
TensorSpecAPI.crystalMlipModels()
    .then((info) => {
        if (!info.installed && dom.stStatus) {
            setStackStatus("MLIP packages not installed yet — stack/CIF/push still work. For relax: pip install matgl ase torch");
        }
    })
    .catch(() => {});

renderLayerRows();
syncExfoliateMode();
refreshExfoliateSources();

TensorSpecAPI.health()
    .then((info) => {
        dom.badge.textContent = `Connected \u2014 ${info.active_sessions} session(s)`;
    })
    .catch(() => {
        dom.badge.textContent = "Backend unreachable";
    });
