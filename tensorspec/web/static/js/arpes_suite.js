/* ARPES data viewer controller.
 *
 * Owns the dashboard state -- which axes are displayed, where the crosshair
 * sits, how wide the integration window is -- and asks the server for a new
 * slice or new curves whenever that state changes. No arithmetic happens here.
 */

import { ImageViewer } from "/static/js/viewers/viewer_2d.js";
import { LineViewer } from "/static/js/viewers/viewer_1d.js";
import { COLORMAP_NAMES } from "/static/js/viewers/colormaps.js";

const el = (id) => document.getElementById(id);

const dom = {
    datasets: el("av-datasets"),
    demo: el("av-demo"),
    refresh: el("av-refresh"),
    badge: el("av-badge"),
    x: el("av-x"),
    y: el("av-y"),
    mode: el("av-mode"),
    cmap: el("av-cmap"),
    profiles: el("av-profiles"),
    dx: el("av-dx"),
    dy: el("av-dy"),
    orthoOn: el("av-ortho-on"),
    ortho: el("av-ortho"),
    readout: el("av-readout"),
    sliders: el("av-sliders"),
    statShape: el("av-stat-shape"),
    statWindow: el("av-stat-window"),
};

const state = {
    name: null,
    axes: [],
    xIdx: 0,
    yIdx: 1,
    fixed: {},
    crosshair: { x: 0, y: 0 },
    header: null,
};

const image = new ImageViewer(el("av-main"), { onCrosshair: onCrosshairMoved });
const profileY = new LineViewer(el("av-prof-y"), { orientation: "vertical", color: "#f87171" });
const profileX = new LineViewer(el("av-prof-x"), { orientation: "horizontal", color: "#60a5fa" });
const profileOrtho = new LineViewer(el("av-prof-ortho"), { orientation: "horizontal", color: "#4ade80" });

const axisText = (axis) => (axis.unit ? `${axis.label} (${axis.unit})` : axis.label);

function setBadge(text, isError = false) {
    dom.badge.textContent = text;
    dom.badge.style.color = isError ? "#ff6b6b" : "";
}

/* ---- workspace list ---- */

async function refreshDatasets() {
    try {
        const listing = await TensorSpecAPI.listItems();
        const tensors = listing.items.filter((item) => item.type === "Spectroscopy DataTree");
        dom.datasets.innerHTML = "";

        if (!tensors.length) {
            dom.datasets.innerHTML = '<li class="empty-state">No datasets in workspace</li>';
            return;
        }

        tensors.forEach((item) => {
            const li = document.createElement("li");
            li.textContent = item.name;
            li.title = item.dims;
            li.style.cursor = "pointer";
            if (item.name === state.name) li.style.color = "var(--accent, #60a5fa)";
            li.addEventListener("click", () => loadTensor(item.name));
            dom.datasets.appendChild(li);
        });
    } catch (err) {
        dom.datasets.innerHTML = `<li class="empty-state">${err.message}</li>`;
    }
}

/* ---- loading a tensor ---- */

async function loadTensor(name) {
    setBadge(`Loading ${name}\u2026`);
    try {
        const info = await TensorSpecAPI.tensorAxes(name);
        state.name = name;
        state.axes = info.axes;
        state.xIdx = info.default_x;
        state.yIdx = info.default_y;
        state.fixed = { ...info.default_fixed };
        state.crosshair = { x: 0, y: 0 };

        buildAxisPickers();
        buildSliders();
        await refreshSlice({ recenter: true });
        await refreshDatasets();
    } catch (err) {
        setBadge(err.message, true);
    }
}

function buildAxisPickers() {
    [dom.x, dom.y].forEach((select, which) => {
        select.innerHTML = "";
        state.axes.forEach((axis) => {
            const option = document.createElement("option");
            option.value = axis.index;
            option.textContent = axisText(axis);
            select.appendChild(option);
        });
        select.value = which === 0 ? state.xIdx : state.yIdx;
    });

    dom.ortho.innerHTML = "";
    state.axes
        .filter((axis) => axis.index !== state.xIdx && axis.index !== state.yIdx)
        .forEach((axis) => {
            const option = document.createElement("option");
            option.value = axis.index;
            option.textContent = axisText(axis);
            dom.ortho.appendChild(option);
        });

    const hasSpare = dom.ortho.options.length > 0;
    dom.orthoOn.disabled = !hasSpare;
    dom.ortho.disabled = !hasSpare || !dom.orthoOn.checked;
}

/* One slider per dimension that is neither displayed axis. */
function buildSliders() {
    dom.sliders.innerHTML = "";
    const spare = state.axes.filter((a) => a.index !== state.xIdx && a.index !== state.yIdx);

    if (!spare.length) {
        dom.sliders.innerHTML = '<p class="hint">Both dimensions of this tensor are displayed.</p>';
        return;
    }

    spare.forEach((axis) => {
        const index = state.fixed[axis.index] ?? Math.floor(axis.size / 2);
        state.fixed[axis.index] = index;

        const row = document.createElement("div");
        row.className = "form-row";
        row.innerHTML = `
            <label>${axisText(axis)}:</label>
            <div class="inline" style="flex:1">
                <span class="hint" data-value style="min-width:8ch">${valueAt(axis, index)}</span>
                <input type="range" min="0" max="${axis.size - 1}" value="${index}" style="flex:1">
                <span class="hint">${index + 1}/${axis.size}</span>
            </div>`;

        const slider = row.querySelector("input");
        const readout = row.querySelector("[data-value]");
        const counter = row.querySelectorAll(".hint")[1];

        slider.addEventListener("input", () => {
            const value = Number(slider.value);
            state.fixed[axis.index] = value;
            readout.textContent = valueAt(axis, value);
            counter.textContent = `${value + 1}/${axis.size}`;
            scheduleSlice();
        });

        dom.sliders.appendChild(row);
    });
}

function valueAt(axis, index) {
    const step = axis.size > 1 ? (axis.max - axis.min) / (axis.size - 1) : 0;
    return (axis.min + step * index).toFixed(3);
}

/* ---- fetching ---- */

let slicePending = null;

/* Dragging a slider fires continuously; coalesce so only the latest
   position reaches the server. */
function scheduleSlice() {
    if (slicePending) clearTimeout(slicePending);
    slicePending = setTimeout(() => { slicePending = null; refreshSlice(); }, 60);
}

async function refreshSlice({ recenter = false } = {}) {
    if (!state.name) return;
    try {
        const { header, values } = await TensorSpecAPI.tensorSlice(state.name, {
            x_idx: state.xIdx,
            y_idx: state.yIdx,
            fixed: state.fixed,
        });

        state.header = header;
        if (recenter) {
            state.crosshair = {
                x: Math.floor(header.shape[1] / 2),
                y: Math.floor(header.shape[0] / 2),
            };
            image.setCrosshair(state.crosshair.x, state.crosshair.y);
        }

        image.setData(header, values);
        image.setWindow(Number(dom.dx.value) || 0, Number(dom.dy.value) || 0);

        const stride = header.stride[0] > 1 || header.stride[1] > 1
            ? ` (decimated ${header.stride[0]}x${header.stride[1]} from ${header.full_shape.join("x")})`
            : "";
        dom.statShape.textContent = `${header.shape[1]} x ${header.shape[0]}${stride}`;
        setBadge(state.name);

        await refreshProfiles();
    } catch (err) {
        setBadge(err.message, true);
    }
}

async function refreshProfiles() {
    if (!state.name || !state.header) return;
    try {
        const payload = {
            x_idx: state.xIdx,
            y_idx: state.yIdx,
            fixed: state.fixed,
            x_center: state.crosshair.x,
            y_center: state.crosshair.y,
            dx: Number(dom.dx.value) || 0,
            dy: Number(dom.dy.value) || 0,
            mode: dom.mode.value,
        };
        if (dom.orthoOn.checked && dom.ortho.value !== "") {
            payload.ortho_idx = Number(dom.ortho.value);
        }

        const result = await TensorSpecAPI.tensorProfiles(state.name, payload);

        const xValue = state.header.x_axis[state.crosshair.x];
        const yValue = state.header.y_axis[state.crosshair.y];

        profileX.setCurve(result.x);
        profileX.setMarker(xValue);
        profileY.setCurve(result.y);
        profileY.setMarker(yValue);

        if (result.ortho) {
            profileOrtho.setCurve(result.ortho);
        } else {
            profileOrtho.setCurve({ axis: [], values: [], label: "Orthogonal", unit: "" });
        }

        const w = result.window;
        dom.statWindow.textContent = `window x[${w.x1}:${w.x2}] y[${w.y1}:${w.y2}]`;
        dom.readout.textContent =
            `${state.header.x_label} ${xValue.toFixed(3)}, ${state.header.y_label} ${yValue.toFixed(3)}`;
    } catch (err) {
        setBadge(err.message, true);
    }
}

/* ---- events ---- */

let profilePending = null;

function onCrosshairMoved(x, y) {
    state.crosshair = { x, y };
    if (profilePending) clearTimeout(profilePending);
    profilePending = setTimeout(() => { profilePending = null; refreshProfiles(); }, 40);
}

function onAxisChanged() {
    const nextX = Number(dom.x.value);
    const nextY = Number(dom.y.value);
    if (nextX === nextY) {
        setBadge("X and Y must differ", true);
        return;
    }
    state.xIdx = nextX;
    state.yIdx = nextY;
    delete state.fixed[nextX];
    delete state.fixed[nextY];
    buildAxisPickers();
    buildSliders();
    refreshSlice({ recenter: true });
}

dom.x.addEventListener("change", onAxisChanged);
dom.y.addEventListener("change", onAxisChanged);
dom.mode.addEventListener("change", refreshProfiles);

[dom.dx, dom.dy].forEach((input) =>
    input.addEventListener("input", () => {
        image.setWindow(Number(dom.dx.value) || 0, Number(dom.dy.value) || 0);
        refreshProfiles();
    })
);

dom.orthoOn.addEventListener("change", () => {
    dom.ortho.disabled = !dom.orthoOn.checked || !dom.ortho.options.length;
    refreshProfiles();
});
dom.ortho.addEventListener("change", refreshProfiles);

dom.cmap.addEventListener("change", () => image.setColormap(dom.cmap.value));

dom.refresh.addEventListener("click", refreshDatasets);
dom.demo.addEventListener("click", async () => {
    dom.demo.disabled = true;
    try {
        await TensorSpecAPI.seedDemo({});
        await refreshDatasets();
        await loadTensor("demo_arpes_cube");
    } catch (err) {
        setBadge(err.message, true);
    } finally {
        dom.demo.disabled = false;
    }
});

COLORMAP_NAMES.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    dom.cmap.appendChild(option);
});

refreshDatasets();
