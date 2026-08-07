/* Crystal Suite controller: binds the Tab 1 controls to the API and viewer.
   Holds no geometry logic; it only moves values between the DOM and the server. */

import { CrystalViewer, elementColor } from "/static/js/viewers/viewer_3d.js";

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
    statAtoms: el("cr-stat-atoms"),
    statBonds: el("cr-stat-bonds"),
};

let viewer = null;
let activeCrystal = null;

function setStatus(message, isError = false) {
    dom.status.textContent = message;
    dom.status.style.color = isError ? "#ff6b6b" : "";
}

function ensureViewer() {
    if (viewer) return viewer;
    dom.placeholder.remove();
    viewer = new CrystalViewer(dom.viewport);
    return viewer;
}

function geometryRequest() {
    return {
        nx: Number(dom.nx.value) || 1,
        ny: Number(dom.ny.value) || 1,
        nz: Number(dom.nz.value) || 1,
        bond_threshold: Number(dom.threshold.value) || 1.15,
        show_bonds: true,
    };
}

async function refreshGeometry({ frame = true } = {}) {
    if (!activeCrystal) return;

    setStatus(`Building geometry for ${activeCrystal}\u2026`);
    try {
        const geometry = await TensorSpecAPI.crystalGeometry(activeCrystal, geometryRequest());
        const view = ensureViewer();
        view.atomScale = Number(dom.radius.value) || 0.5;
        view.render(geometry, { frame });

        dom.statAtoms.textContent = `${geometry.n_atoms} atoms`;
        dom.statBonds.textContent = `${geometry.bonds.length} bonds`;
        dom.legend.textContent = geometry.elements.join(", ");
        dom.legend.style.color = elementColor(geometry.elements[0]);
        setStatus(`${activeCrystal} rendered`);
    } catch (err) {
        setStatus(err.message, true);
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
    } catch (err) {
        setStatus(err.message, true);
    } finally {
        // Clear so re-picking the same file still fires a change event.
        event.target.value = "";
    }
}

dom.load.addEventListener("click", () => dom.file.click());
dom.file.addEventListener("change", onFileChosen);

[dom.nx, dom.ny, dom.nz, dom.threshold].forEach((input) =>
    input.addEventListener("change", () => refreshGeometry({ frame: true }))
);

dom.radius.addEventListener("change", () => {
    if (viewer) viewer.setAtomScale(Number(dom.radius.value) || 0.5);
});

TensorSpecAPI.health()
    .then((info) => {
        dom.badge.textContent = `Connected \u2014 ${info.active_sessions} session(s)`;
    })
    .catch(() => {
        dom.badge.textContent = "Backend unreachable";
    });
