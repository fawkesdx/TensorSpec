/* ARPES Suite: snap-grid Data Viewer + Option A/B1 simulator.
 *
 * Panels ask the server for slices/profiles; the simulator queues jobs and
 * streams logs. No physics arithmetic lives here.
 */

import { ImageViewer } from "/static/js/viewers/viewer_2d.js";
import { LineViewer } from "/static/js/viewers/viewer_1d.js";
import { COLORMAP_NAMES } from "/static/js/viewers/colormaps.js";

const el = (id) => document.getElementById(id);
const axisText = (axis) => (axis.unit ? `${axis.label} (${axis.unit})` : axis.label);

function nearestIndex(axis, value) {
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < axis.length; i++) {
        const dist = Math.abs(axis[i] - value);
        if (dist < bestDist) {
            bestDist = dist;
            best = i;
        }
    }
    return best;
}

function downloadBlob(filename, blob) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

function downloadText(filename, text, mime = "text/csv") {
    downloadBlob(filename, new Blob([text], { type: `${mime};charset=utf-8` }));
}

function csvEscape(value) {
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

const syncBus = {
    panels: new Set(),
    broadcast(source, xLabel, xValue, yLabel, yValue) {
        for (const panel of this.panels) {
            if (panel === source || !panel.syncEnabled) continue;
            panel.receiveSync(xLabel, xValue, yLabel, yValue);
        }
    },
};

/* ---- Snap-grid Data Viewer ---- */

class ViewerPanel {
    constructor(board, { row = 0, col = 0 } = {}) {
        this.board = board;
        this.gridRow = row;
        this.gridCol = col;
        this.detached = false;
        this.floatEl = null;

        this.name = null;
        this.axes = [];
        this.xIdx = 0;
        this.yIdx = 1;
        this.fixed = {};
        this.crosshair = { x: 0, y: 0 };
        this.header = null;
        this.lastProfiles = null;
        this.syncing = false;
        this.slicePending = null;
        this.profilePending = null;

        this.root = document.createElement("article");
        this.root.className = "av-panel";
        this.root.innerHTML = `
            <div class="av-panel__toolbar" data-toolbar>
                <span class="av-panel__title" data-title title="Right-click for snap menu">Empty panel</span>
                <label class="check" style="margin:0"><input type="checkbox" data-sync> Sync</label>
                <button type="button" class="btn" data-csv>CSV</button>
                <button type="button" class="btn" data-pdf>PDF</button>
                <button type="button" class="btn" data-svg>SVG</button>
                <button type="button" class="btn" data-png>PNG</button>
                <button type="button" class="btn" data-close>X</button>
            </div>
            <div class="inline" style="margin-bottom:8px">
                <label>X:</label><select class="field" data-x style="width:auto"></select>
                <label>Y:</label><select class="field" data-y style="width:auto"></select>
                <label class="check" style="margin:0"><input type="checkbox" data-profiles checked> XY Profiles</label>
                <select class="field" data-mode style="width:auto">
                    <option value="sum">Raw (Sum)</option>
                    <option value="mean">Mean</option>
                    <option value="normalized">Normalized to Max</option>
                </select>
                <select class="field" data-cmap style="width:auto"></select>
            </div>
            <div class="inline" style="margin-bottom:8px">
                <label>&#916;X:</label><input class="field field--num" data-dx type="number" min="0" max="100" value="0">
                <label>&#916;Y:</label><input class="field field--num" data-dy type="number" min="0" max="100" value="0">
                <label class="check" style="margin:0"><input type="checkbox" data-ortho-on> Ortho:</label>
                <select class="field" data-ortho style="width:auto" disabled></select>
                <span class="hint" data-readout style="margin:0">&mdash;</span>
            </div>
            <div class="viewer-grid">
                <div class="viewport" data-main></div>
                <div class="viewport" data-prof-y></div>
                <div class="viewport" data-prof-x></div>
                <div class="viewport" data-prof-ortho></div>
            </div>
            <h3 class="section-heading">Dimension Sliders</h3>
            <div data-sliders><p class="hint">Load a dataset to begin.</p></div>
            <p class="hint" data-window style="margin-top:8px">&mdash;</p>`;

        this.dom = {
            title: this.root.querySelector("[data-title]"),
            toolbar: this.root.querySelector("[data-toolbar]"),
            sync: this.root.querySelector("[data-sync]"),
            csv: this.root.querySelector("[data-csv]"),
            pdf: this.root.querySelector("[data-pdf]"),
            svg: this.root.querySelector("[data-svg]"),
            png: this.root.querySelector("[data-png]"),
            close: this.root.querySelector("[data-close]"),
            x: this.root.querySelector("[data-x]"),
            y: this.root.querySelector("[data-y]"),
            profiles: this.root.querySelector("[data-profiles]"),
            mode: this.root.querySelector("[data-mode]"),
            cmap: this.root.querySelector("[data-cmap]"),
            dx: this.root.querySelector("[data-dx]"),
            dy: this.root.querySelector("[data-dy]"),
            orthoOn: this.root.querySelector("[data-ortho-on]"),
            ortho: this.root.querySelector("[data-ortho]"),
            readout: this.root.querySelector("[data-readout]"),
            sliders: this.root.querySelector("[data-sliders]"),
            window: this.root.querySelector("[data-window]"),
            main: this.root.querySelector("[data-main]"),
            profY: this.root.querySelector("[data-prof-y]"),
            profX: this.root.querySelector("[data-prof-x]"),
            profOrtho: this.root.querySelector("[data-prof-ortho]"),
        };

        COLORMAP_NAMES.forEach((name) => {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            this.dom.cmap.appendChild(option);
        });

        this.image = new ImageViewer(this.dom.main, { onCrosshair: (x, y) => this.onCrosshairMoved(x, y) });
        this.profileY = new LineViewer(this.dom.profY, { orientation: "vertical", color: "#f87171" });
        this.profileX = new LineViewer(this.dom.profX, { orientation: "horizontal", color: "#60a5fa" });
        this.profileOrtho = new LineViewer(this.dom.profOrtho, { orientation: "horizontal", color: "#4ade80" });

        this.root.addEventListener("pointerdown", () => this.board.setActive(this));
        this.dom.toolbar.addEventListener("contextmenu", (event) => {
            event.preventDefault();
            this.board.openMenu(this, event.clientX, event.clientY);
        });
        this.dom.close.addEventListener("click", () => this.board.closePanel(this));
        this.dom.csv.addEventListener("click", () => this.exportCsv());
        this.dom.pdf.addEventListener("click", () => this.exportFigure("pdf"));
        this.dom.svg.addEventListener("click", () => this.exportFigure("svg"));
        this.dom.png.addEventListener("click", () => this.exportClientPng());
        this.dom.x.addEventListener("change", () => this.onAxisChanged());
        this.dom.y.addEventListener("change", () => this.onAxisChanged());
        this.dom.mode.addEventListener("change", () => this.refreshProfiles());
        this.dom.cmap.addEventListener("change", () => this.image.setColormap(this.dom.cmap.value));
        [this.dom.dx, this.dom.dy].forEach((input) =>
            input.addEventListener("input", () => {
                this.image.setWindow(Number(this.dom.dx.value) || 0, Number(this.dom.dy.value) || 0);
                this.refreshProfiles();
            })
        );
        this.dom.orthoOn.addEventListener("change", () => {
            this.dom.ortho.disabled = !this.dom.orthoOn.checked || !this.dom.ortho.options.length;
            this.refreshProfiles();
        });
        this.dom.ortho.addEventListener("change", () => this.refreshProfiles());

        syncBus.panels.add(this);
    }

    get syncEnabled() {
        return this.dom.sync.checked;
    }

    setActive(active) {
        this.root.classList.toggle("av-panel--active", active);
    }

    applyGridPlacement() {
        this.root.style.gridRow = String(this.gridRow + 1);
        this.root.style.gridColumn = String(this.gridCol + 1);
    }

    destroy() {
        syncBus.panels.delete(this);
        if (this.floatEl) this.floatEl.remove();
        this.root.remove();
    }

    displayedAxisKey(which) {
        const idx = which === "x" ? this.xIdx : this.yIdx;
        const axis = this.axes.find((a) => a.index === idx);
        return axis ? axisText(axis) : "";
    }

    async loadTensor(name) {
        const info = await TensorSpecAPI.tensorAxes(name);
        this.name = name;
        this.axes = info.axes;
        this.xIdx = info.default_x;
        this.yIdx = info.default_y;
        this.fixed = { ...info.default_fixed };
        this.crosshair = { x: 0, y: 0 };
        this.dom.title.textContent = name;
        this.buildAxisPickers();
        this.buildSliders();
        await this.refreshSlice({ recenter: true });
    }

    buildAxisPickers() {
        [this.dom.x, this.dom.y].forEach((select, which) => {
            select.innerHTML = "";
            this.axes.forEach((axis) => {
                const option = document.createElement("option");
                option.value = axis.index;
                option.textContent = axisText(axis);
                select.appendChild(option);
            });
            select.value = which === 0 ? this.xIdx : this.yIdx;
        });
        this.dom.ortho.innerHTML = "";
        this.axes
            .filter((axis) => axis.index !== this.xIdx && axis.index !== this.yIdx)
            .forEach((axis) => {
                const option = document.createElement("option");
                option.value = axis.index;
                option.textContent = axisText(axis);
                this.dom.ortho.appendChild(option);
            });
        const hasSpare = this.dom.ortho.options.length > 0;
        this.dom.orthoOn.disabled = !hasSpare;
        this.dom.ortho.disabled = !hasSpare || !this.dom.orthoOn.checked;
    }

    buildSliders() {
        this.dom.sliders.innerHTML = "";
        const spare = this.axes.filter((a) => a.index !== this.xIdx && a.index !== this.yIdx);
        if (!spare.length) {
            this.dom.sliders.innerHTML = '<p class="hint">Both dimensions displayed.</p>';
            return;
        }
        spare.forEach((axis) => {
            const index = this.fixed[axis.index] ?? Math.floor(axis.size / 2);
            this.fixed[axis.index] = index;
            const row = document.createElement("div");
            row.className = "form-row";
            const step = axis.size > 1 ? (axis.max - axis.min) / (axis.size - 1) : 0;
            const value = (axis.min + step * index).toFixed(3);
            row.innerHTML = `
                <label>${axisText(axis)}:</label>
                <div class="inline" style="flex:1">
                    <span class="hint" data-value style="min-width:8ch">${value}</span>
                    <input type="range" min="0" max="${axis.size - 1}" value="${index}" style="flex:1">
                    <span class="hint">${index + 1}/${axis.size}</span>
                </div>`;
            const slider = row.querySelector("input");
            const readout = row.querySelector("[data-value]");
            const counter = row.querySelectorAll(".hint")[1];
            slider.addEventListener("input", () => {
                const v = Number(slider.value);
                this.fixed[axis.index] = v;
                readout.textContent = (axis.min + step * v).toFixed(3);
                counter.textContent = `${v + 1}/${axis.size}`;
                this.scheduleSlice();
            });
            this.dom.sliders.appendChild(row);
        });
    }

    scheduleSlice() {
        if (this.slicePending) clearTimeout(this.slicePending);
        this.slicePending = setTimeout(() => {
            this.slicePending = null;
            this.refreshSlice();
        }, 60);
    }

    async refreshSlice({ recenter = false } = {}) {
        if (!this.name) return;
        const { header, values } = await TensorSpecAPI.tensorSlice(this.name, {
            x_idx: this.xIdx,
            y_idx: this.yIdx,
            fixed: this.fixed,
        });
        this.header = header;
        if (recenter) {
            this.crosshair = {
                x: Math.floor(header.shape[1] / 2),
                y: Math.floor(header.shape[0] / 2),
            };
            this.image.setCrosshair(this.crosshair.x, this.crosshair.y);
        }
        this.image.setData(header, values);
        this.image.setWindow(Number(this.dom.dx.value) || 0, Number(this.dom.dy.value) || 0);
        await this.refreshProfiles();
        this.board.updateStatus(this);
    }

    async refreshProfiles() {
        if (!this.name || !this.header) return;
        const payload = {
            x_idx: this.xIdx,
            y_idx: this.yIdx,
            fixed: this.fixed,
            x_center: this.crosshair.x,
            y_center: this.crosshair.y,
            dx: Number(this.dom.dx.value) || 0,
            dy: Number(this.dom.dy.value) || 0,
            mode: this.dom.mode.value,
        };
        if (this.dom.orthoOn.checked && this.dom.ortho.value !== "") {
            payload.ortho_idx = Number(this.dom.ortho.value);
        }
        const result = await TensorSpecAPI.tensorProfiles(this.name, payload);
        this.lastProfiles = result;
        const xValue = this.header.x_axis[this.crosshair.x];
        const yValue = this.header.y_axis[this.crosshair.y];
        this.profileX.setCurve(result.x);
        this.profileX.setMarker(xValue);
        this.profileY.setCurve(result.y);
        this.profileY.setMarker(yValue);
        this.profileOrtho.setCurve(
            result.ortho || { axis: [], values: [], label: "Orthogonal", unit: "" }
        );
        const w = result.window;
        this.dom.window.textContent = `window x[${w.x1}:${w.x2}] y[${w.y1}:${w.y2}]`;
        this.dom.readout.textContent =
            `${this.header.x_label} ${xValue.toFixed(3)}, ${this.header.y_label} ${yValue.toFixed(3)}`;
        this.board.updateStatus(this);
    }

    onCrosshairMoved(x, y) {
        this.crosshair = { x, y };
        if (this.profilePending) clearTimeout(this.profilePending);
        this.profilePending = setTimeout(() => {
            this.profilePending = null;
            this.refreshProfiles();
        }, 40);
        if (this.syncEnabled && this.header && !this.syncing) {
            syncBus.broadcast(
                this,
                this.displayedAxisKey("x"),
                this.header.x_axis[x],
                this.displayedAxisKey("y"),
                this.header.y_axis[y]
            );
        }
    }

    receiveSync(sourceXLabel, sourceXValue, sourceYLabel, sourceYValue) {
        if (!this.header || this.syncing) return;
        this.syncing = true;
        let nextX = this.crosshair.x;
        let nextY = this.crosshair.y;
        let changed = false;
        const myX = this.displayedAxisKey("x");
        const myY = this.displayedAxisKey("y");
        if (myX === sourceXLabel) {
            nextX = nearestIndex(this.header.x_axis, sourceXValue);
            changed = true;
        } else if (myX === sourceYLabel) {
            nextX = nearestIndex(this.header.x_axis, sourceYValue);
            changed = true;
        }
        if (myY === sourceYLabel) {
            nextY = nearestIndex(this.header.y_axis, sourceYValue);
            changed = true;
        } else if (myY === sourceXLabel) {
            nextY = nearestIndex(this.header.y_axis, sourceXValue);
            changed = true;
        }
        if (changed && (nextX !== this.crosshair.x || nextY !== this.crosshair.y)) {
            this.crosshair = { x: nextX, y: nextY };
            this.image.setCrosshair(nextX, nextY);
            this.refreshProfiles();
        }
        this.syncing = false;
    }

    onAxisChanged() {
        const nextX = Number(this.dom.x.value);
        const nextY = Number(this.dom.y.value);
        if (nextX === nextY) {
            this.board.setBadge("X and Y must differ", true);
            return;
        }
        this.xIdx = nextX;
        this.yIdx = nextY;
        delete this.fixed[nextX];
        delete this.fixed[nextY];
        this.buildAxisPickers();
        this.buildSliders();
        this.refreshSlice({ recenter: true });
    }

    exportCsv() {
        if (!this.dom.profiles.checked) {
            this.board.setBadge("Enable XY Profiles before exporting.", true);
            return;
        }
        if (!this.lastProfiles || !this.header) {
            this.board.setBadge("No profiles to export yet.", true);
            return;
        }
        const { x, y, ortho } = this.lastProfiles;
        const headers = [
            `Horizontal_${this.header.x_label}_${this.header.x_unit || "au"}`,
            `Horizontal_${this.header.x_label}_Intensity`,
            `Vertical_${this.header.y_label}_${this.header.y_unit || "au"}`,
            `Vertical_${this.header.y_label}_Intensity`,
        ];
        const columns = [x.axis, x.values, y.axis, y.values];
        if (ortho?.axis?.length) {
            headers.push(`Ortho_${ortho.label}_${ortho.unit || "au"}`, `Ortho_${ortho.label}_Intensity`);
            columns.push(ortho.axis, ortho.values);
        }
        const maxLen = Math.max(...columns.map((c) => c.length));
        const lines = [headers.map(csvEscape).join(",")];
        for (let i = 0; i < maxLen; i++) {
            lines.push(columns.map((c) => (i < c.length ? c[i] : "")).join(","));
        }
        const safe = (this.name || "profiles").replace(/[^\w.-]+/g, "_");
        downloadText(`${safe}_profiles.csv`, `${lines.join("\n")}\n`);
        this.board.setBadge(`Exported ${safe}_profiles.csv`);
    }

    async exportFigure(fmt) {
        if (!this.name || !this.header) {
            this.board.setBadge("Load a dataset first.", true);
            return;
        }
        try {
            const blob = await TensorSpecAPI.arpesExportFigure(this.name, {
                x_idx: this.xIdx,
                y_idx: this.yIdx,
                fixed: this.fixed,
                x_center: this.crosshair.x,
                y_center: this.crosshair.y,
                dx: Number(this.dom.dx.value) || 0,
                dy: Number(this.dom.dy.value) || 0,
                mode: this.dom.mode.value,
                include_profiles: this.dom.profiles.checked,
                fmt,
                title: this.name,
            });
            downloadBlob(`${this.name}_figure.${fmt}`, blob);
            this.board.setBadge(`Exported ${this.name}_figure.${fmt}`);
        } catch (err) {
            this.board.setBadge(err.message, true);
        }
    }

    exportClientPng() {
        const canvases = [this.dom.main, this.dom.profY, this.dom.profX, this.dom.profOrtho]
            .map((node) => node.querySelector("canvas"))
            .filter(Boolean);
        if (!canvases.length) {
            this.board.setBadge("Nothing to capture yet.", true);
            return;
        }
        const main = canvases[0];
        const width = Math.max(main.width, 800);
        const height = Math.max(main.height + 160, 600);
        const out = document.createElement("canvas");
        out.width = width;
        out.height = height;
        const ctx = out.getContext("2d");
        ctx.fillStyle = "#12121a";
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(main, 0, 0);
        if (canvases[2]) ctx.drawImage(canvases[2], 0, main.height, width * 0.75, 140);
        if (canvases[1]) ctx.drawImage(canvases[1], width * 0.75, 0, width * 0.25, main.height);
        out.toBlob((blob) => {
            if (!blob) return;
            downloadBlob(`${this.name || "panel"}_client.png`, blob);
            this.board.setBadge("Exported client PNG");
        });
    }
}

class SnapBoard {
    constructor(host) {
        this.host = host;
        this.panels = [];
        this.active = null;
        this.menu = null;
        this.detachedHost = document.createElement("div");
        this.detachedHost.className = "av-float-host";
        document.body.appendChild(this.detachedHost);
        document.addEventListener("click", () => this.closeMenu());
    }

    setBadge(text, isError = false) {
        const badge = el("av-badge");
        badge.textContent = text;
        badge.style.color = isError ? "#ff6b6b" : "";
    }

    setActive(panel) {
        this.active = panel;
        this.panels.forEach((p) => p.setActive(p === panel));
        if (panel?.name) this.setBadge(panel.name);
    }

    updateStatus(panel) {
        if (panel !== this.active || !panel.header) return;
        const h = panel.header;
        const stride = h.stride[0] > 1 || h.stride[1] > 1
            ? ` (decimated ${h.stride[0]}x${h.stride[1]})`
            : "";
        el("av-stat-shape").textContent = `${h.shape[1]} x ${h.shape[0]}${stride}`;
        el("av-stat-window").textContent = panel.dom.window.textContent;
    }

    rebuild() {
        const attached = this.panels.filter((p) => !p.detached);
        let maxR = 0;
        let maxC = 0;
        attached.forEach((p) => {
            maxR = Math.max(maxR, p.gridRow);
            maxC = Math.max(maxC, p.gridCol);
        });
        this.host.style.gridTemplateRows = `repeat(${maxR + 1}, minmax(320px, 1fr))`;
        this.host.style.gridTemplateColumns = `repeat(${maxC + 1}, minmax(420px, 1fr))`;
        attached.forEach((p) => {
            if (p.root.parentElement !== this.host) this.host.appendChild(p.root);
            p.applyGridPlacement();
        });
    }

    spawn(ref, direction) {
        let row = 0;
        let col = 0;
        if (ref && direction) {
            row = ref.gridRow;
            col = ref.gridCol;
            if (direction === "Top") row -= 1;
            if (direction === "Bottom") row += 1;
            if (direction === "Left") col -= 1;
            if (direction === "Right") col += 1;
            if (row < 0) {
                this.panels.forEach((p) => {
                    if (!p.detached) p.gridRow += 1;
                });
                row = 0;
            }
            if (col < 0) {
                this.panels.forEach((p) => {
                    if (!p.detached) p.gridCol += 1;
                });
                col = 0;
            }
        } else if (ref) {
            row = ref.gridRow + 1;
            col = ref.gridCol;
        }
        const panel = new ViewerPanel(this, { row, col });
        this.panels.push(panel);
        this.host.appendChild(panel.root);
        this.rebuild();
        this.setActive(panel);
        return panel;
    }

    neighbor(panel, direction) {
        let r = panel.gridRow;
        let c = panel.gridCol;
        if (direction === "up") r -= 1;
        if (direction === "down") r += 1;
        if (direction === "left") c -= 1;
        if (direction === "right") c += 1;
        return this.panels.find((p) => !p.detached && p.gridRow === r && p.gridCol === c) || null;
    }

    closePanel(panel) {
        const attached = this.panels.filter((p) => !p.detached);
        if (attached.length <= 1 && !panel.detached) {
            this.setBadge("Cannot close the last panel.", true);
            return;
        }
        const idx = this.panels.indexOf(panel);
        panel.destroy();
        this.panels.splice(idx, 1);
        this.rebuild();
        this.setActive(this.panels[0] || null);
    }

    detach(panel) {
        const attached = this.panels.filter((p) => !p.detached);
        if (attached.length <= 1) {
            this.setBadge("Cannot detach the last panel.", true);
            return;
        }
        panel.detached = true;
        const float = document.createElement("div");
        float.className = "av-float";
        float.innerHTML = `<div class="av-float__bar"><span>Detached</span><button type="button" class="btn" data-reattach>Reattach</button></div>`;
        float.querySelector("[data-reattach]").addEventListener("click", () => this.reattach(panel));
        float.appendChild(panel.root);
        this.detachedHost.appendChild(float);
        panel.floatEl = float;
        this.rebuild();
    }

    reattach(panel) {
        const attached = this.panels.filter((p) => !p.detached);
        const maxRow = attached.reduce((m, p) => Math.max(m, p.gridRow), -1);
        panel.detached = false;
        panel.gridRow = maxRow + 1;
        panel.gridCol = 0;
        if (panel.floatEl) {
            panel.floatEl.remove();
            panel.floatEl = null;
        }
        this.host.appendChild(panel.root);
        this.rebuild();
        this.setActive(panel);
    }

    openMenu(panel, x, y) {
        this.closeMenu();
        const menu = document.createElement("div");
        menu.className = "av-menu";
        menu.style.left = `${x}px`;
        menu.style.top = `${y}px`;
        const items = [
            ["Spawn Up", () => this.spawn(panel, "Top")],
            ["Spawn Down", () => this.spawn(panel, "Bottom")],
            ["Spawn Left", () => this.spawn(panel, "Left")],
            ["Spawn Right", () => this.spawn(panel, "Right")],
            ["Detach", () => this.detach(panel)],
            ["Close", () => this.closePanel(panel)],
        ];
        ["up", "down", "left", "right"].forEach((dir) => {
            const n = this.neighbor(panel, dir);
            if (n) items.splice(4, 0, [`De-snap ${dir}`, () => this.detach(n)]);
        });
        if (panel.detached) {
            items.length = 0;
            items.push(["Reattach", () => this.reattach(panel)]);
            items.push(["Close", () => this.closePanel(panel)]);
        }
        items.forEach(([label, action]) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.textContent = label;
            btn.addEventListener("click", (event) => {
                event.stopPropagation();
                this.closeMenu();
                action();
            });
            menu.appendChild(btn);
        });
        document.body.appendChild(menu);
        this.menu = menu;
    }

    closeMenu() {
        if (this.menu) {
            this.menu.remove();
            this.menu = null;
        }
    }
}

/* ---- Simulator ---- */

const sim = {
    jobId: null,
    eAxis: null,
    socket: null,
    viewer: null,
};

function appendLog(line) {
    const box = el("ar-log");
    if (box.textContent === "Waiting to run…") box.textContent = "";
    box.textContent += `${line}\n`;
    box.scrollTop = box.scrollHeight;
}

async function refreshCrystals() {
    const listing = await TensorSpecAPI.listItems();
    const crystals = listing.items.filter((item) =>
        /crystal/i.test(item.type) || /structure/i.test(item.type)
    );
    const select = el("ar-crystal");
    const previous = select.value;
    select.innerHTML = "";
    if (!crystals.length) {
        select.innerHTML = '<option value="">No crystals available</option>';
        return;
    }
    crystals.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = `${item.name} — ${item.dims}`;
        select.appendChild(option);
    });
    if ([...select.options].some((o) => o.value === previous)) select.value = previous;
}

function simPayload() {
    return {
        crystal_name: el("ar-crystal").value,
        model: el("ar-model").value,
        store_as: "simulated_arpes",
        photon_energy: Number(el("ar-hv").value),
        work_function: Number(el("ar-phi").value),
        inner_potential: Number(el("ar-v0").value),
        temperature: Number(el("ar-temp").value),
        incidence_angle: Number(el("ar-inc").value),
        polarization: el("ar-pol").value,
        lin_pol_angle: Number(el("ar-polang").value),
        matrix_element_mode: el("ar-int").value,
        manip_theta: Number(el("ar-theta").value),
        manip_azimuth: Number(el("ar-azi").value),
        manip_tilt: Number(el("ar-tilt").value),
        h: Number(el("ar-h").value),
        k: Number(el("ar-k").value),
        l: Number(el("ar-l").value),
        slit_angle: Number(el("ar-slitang").value),
        kx: {
            min: Number(el("ar-kx-min").value),
            max: Number(el("ar-kx-max").value),
            steps: Number(el("ar-kx-pts").value),
        },
        ky: {
            min: Number(el("ar-ky-min").value),
            max: Number(el("ar-ky-max").value),
            steps: Number(el("ar-ky-pts").value),
        },
        energy: {
            min: Number(el("ar-e-min").value),
            max: Number(el("ar-e-max").value),
            steps: Number(el("ar-e-pts").value),
        },
        se_width: Number(el("ar-se").value),
        res_E: Number(el("ar-de").value),
        res_k: Number(el("ar-dk").value),
        mesh_resolution: 16,
    };
}

function watchSimJob(jobId) {
    if (sim.socket) {
        sim.socket.close();
        sim.socket = null;
    }
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${location.host}/api/arpes/jobs/${encodeURIComponent(jobId)}/logs`);
    sim.socket = socket;
    socket.addEventListener("message", async (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "log") appendLog(message.line);
        if (message.type === "status") {
            el("ar-sim-status").textContent = `Job ${message.status}`;
            if (message.status === "succeeded") {
                el("ar-run").disabled = false;
                el("ar-cancel").disabled = true;
                el("ar-push").disabled = false;
                await loadPreview(0);
            }
            if (["failed", "cancelled"].includes(message.status)) {
                el("ar-run").disabled = false;
                el("ar-cancel").disabled = true;
                appendLog(message.error || `Job ${message.status}`);
            }
        }
        if (message.type === "error") appendLog(message.detail || "websocket error");
    });
}

async function loadPreview(eIndex) {
    if (!sim.jobId) return;
    if (!sim.viewer) sim.viewer = new ImageViewer(el("ar-intensity"));
    const { header, values } = await TensorSpecAPI.arpesPreview(sim.jobId, eIndex);
    sim.viewer.setData(header, values);
    const slider = el("ar-be-slider");
    const spin = el("ar-be");
    slider.disabled = false;
    spin.disabled = false;
    slider.max = String((header.n_energy || 1) - 1);
    slider.value = String(header.e_index ?? eIndex);
    spin.value = Number(header.energy || 0).toFixed(3);
    el("ar-sim-status").textContent =
        `Preview E=${Number(header.energy).toFixed(3)} eV (${header.shape[1]}×${header.shape[0]})`;
}

/* ---- boot ---- */

const board = new SnapBoard(el("av-panel-host"));
el("av-panel-host").classList.add("av-panel-host--snap");
board.spawn(null, null);

async function refreshDatasets() {
    try {
        const listing = await TensorSpecAPI.listItems();
        const tensors = listing.items.filter((item) => item.type === "Spectroscopy DataTree");
        const list = el("av-datasets");
        list.innerHTML = "";
        if (!tensors.length) {
            list.innerHTML = '<li class="empty-state">No datasets in workspace</li>';
            return;
        }
        tensors.forEach((item) => {
            const li = document.createElement("li");
            li.textContent = item.name;
            li.style.cursor = "pointer";
            if (board.active?.name === item.name) li.style.color = "var(--accent, #60a5fa)";
            li.addEventListener("click", async () => {
                const target = board.active || board.panels[0];
                board.setActive(target);
                board.setBadge(`Loading ${item.name}\u2026`);
                try {
                    await target.loadTensor(item.name);
                    board.setBadge(item.name);
                    await refreshDatasets();
                } catch (err) {
                    board.setBadge(err.message, true);
                }
            });
            list.appendChild(li);
        });
    } catch (err) {
        el("av-datasets").innerHTML = `<li class="empty-state">${err.message}</li>`;
    }
}

el("av-refresh").addEventListener("click", refreshDatasets);
el("av-add-panel").addEventListener("click", () => {
    board.spawn(board.active || board.panels[0], "Bottom");
    board.setBadge("Spawned panel — click a dataset to load");
});
el("av-demo").addEventListener("click", async () => {
    el("av-demo").disabled = true;
    try {
        await TensorSpecAPI.seedDemo({});
        await refreshDatasets();
        const target = board.active || board.spawn(null, null);
        await target.loadTensor("demo_arpes_cube");
        board.setActive(target);
        board.setBadge("demo_arpes_cube");
    } catch (err) {
        board.setBadge(err.message, true);
    } finally {
        el("av-demo").disabled = false;
    }
});

el("av-load-file").addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    event.target.value = "";
    if (!file) return;
    const logInput = el("av-load-log");
    const logFile = logInput?.files && logInput.files[0] ? logInput.files[0] : null;
    board.setBadge(`Loading ${file.name}\u2026`);
    try {
        const result = await TensorSpecAPI.loadArpes(file, { logFile });
        await refreshDatasets();
        const target = board.active || board.spawn(null, null);
        await target.loadTensor(result.name);
        board.setActive(target);
        const kind = result.measurement_type || result.data_type || "ARPES";
        board.setBadge(`${result.name} · ${kind} · ${result.shape.join("\u00d7")}`);
    } catch (err) {
        board.setBadge(err.message, true);
    }
});

el("ar-crystal-refresh").addEventListener("click", () => refreshCrystals().catch((e) => appendLog(e.message)));
el("ar-run").addEventListener("click", async () => {
    if (!el("ar-crystal").value) {
        appendLog("Select a crystal first (Crystal Suite → load CIF → it appears here).");
        return;
    }
    el("ar-run").disabled = true;
    el("ar-cancel").disabled = false;
    el("ar-push").disabled = true;
    el("ar-log").textContent = "";
    try {
        const job = await TensorSpecAPI.arpesSimulate(simPayload());
        sim.jobId = job.job_id;
        appendLog(`Queued ${job.job_id}`);
        watchSimJob(job.job_id);
    } catch (err) {
        appendLog(err.message);
        el("ar-run").disabled = false;
        el("ar-cancel").disabled = true;
    }
});
el("ar-cancel").addEventListener("click", async () => {
    if (!sim.jobId) return;
    try {
        await TensorSpecAPI.arpesCancelJob(sim.jobId);
        appendLog("Cancel requested");
    } catch (err) {
        appendLog(err.message);
    }
});
el("ar-push").addEventListener("click", async () => {
    if (!sim.jobId) return;
    try {
        const result = await TensorSpecAPI.arpesPushJob(sim.jobId, {});
        appendLog(`Pushed ${result.name} ${JSON.stringify(result.shape)}`);
        await refreshDatasets();
        el("ar-sim-status").textContent = `Pushed as ${result.name} — open Data Viewer`;
    } catch (err) {
        appendLog(err.message);
    }
});
el("ar-be-slider").addEventListener("input", () => {
    loadPreview(Number(el("ar-be-slider").value)).catch((err) => appendLog(err.message));
});

refreshDatasets();
refreshCrystals().catch(() => {});

/* ---- Process tab: in-plane → k∥ ---- */

const processState = {
    roles: null,
    rawAxes: null,
    rawHeader: null,
    fixedAxis: null,
    bzPolygon: null,
    perpBz: null,
    rawViewer: null,
    kViewer: null,
    debounce: null,
    mode: "inplane",
};

function processMode() {
    return el("ap-mode")?.value || "inplane";
}

function syncProcessModeUI() {
    const mode = processMode();
    processState.mode = mode;
    const kz = mode === "kz";
    el("ap-inplane-controls").hidden = kz;
    el("ap-kz-controls").hidden = !kz;
    el("ap-bz-inplane-row").hidden = kz;
    el("ap-bz-inplane-row").style.display = kz ? "none" : "flex";
    el("ap-bz-perp-row").hidden = !kz;
    el("ap-bz-perp-row").style.display = kz ? "flex" : "none";
    el("ap-raw-title").textContent = kz ? "Raw (hv scan)" : "Raw (set Γ)";
    el("ap-k-title").textContent = kz ? "kz preview" : "k∥ preview";
    el("ap-status").textContent = kz
        ? "Drag the Vo slider — kz preview updates live. Toggle ⊥ BZ to judge Vo."
        : "Click the raw map to set Γ. Preview updates on the right.";
}

function processEnsureViewers() {
    if (!processState.rawViewer) {
        processState.rawViewer = new ImageViewer(el("ap-raw"), {
            onCrosshair: (xi, yi) => {
                if (!processState.rawHeader || !processState.roles) return;
                const roles = processState.roles;
                const xIdx = processState.viewX;
                const yIdx = processState.viewY;
                const xAxis = processState.rawHeader.x_axis;
                const yAxis = processState.rawHeader.y_axis;
                if (xIdx === roles.angle_axis) {
                    el("ap-center").value = String(xAxis[xi]);
                } else if (yIdx === roles.angle_axis) {
                    el("ap-center").value = String(yAxis[yi]);
                }
                if (roles.beta_axis != null) {
                    if (xIdx === roles.beta_axis) {
                        el("ap-beta-center").value = String(xAxis[xi]);
                    } else if (yIdx === roles.beta_axis) {
                        el("ap-beta-center").value = String(yAxis[yi]);
                    }
                }
                scheduleProcessPreview();
            },
        });
    }
    if (!processState.kViewer) {
        processState.kViewer = new ImageViewer(el("ap-k"));
    }
}

async function refreshProcessDatasets() {
    const listing = await TensorSpecAPI.listItems();
    const tensors = listing.items.filter((item) => item.type === "Spectroscopy DataTree");
    const select = el("ap-dataset");
    const previous = select.value;
    select.innerHTML = "";
    if (!tensors.length) {
        select.innerHTML = '<option value="">No spectroscopy data</option>';
        return;
    }
    tensors.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = item.name;
        select.appendChild(option);
    });
    if ([...select.options].some((o) => o.value === previous)) select.value = previous;
}

async function refreshProcessCrystals() {
    const listing = await TensorSpecAPI.listItems();
    const crystals = listing.items.filter(
        (item) => /crystal/i.test(item.type) || /structure/i.test(item.type)
    );
    const select = el("ap-crystal");
    const previous = select.value;
    select.innerHTML = "";
    if (!crystals.length) {
        select.innerHTML = '<option value="">No crystal</option>';
        return;
    }
    crystals.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = item.name;
        select.appendChild(option);
    });
    if ([...select.options].some((o) => o.value === previous)) select.value = previous;
}

function processViewAxes(roles) {
    const mode = processMode();
    if (mode === "kz") {
        if (roles.photon_axis == null) {
            return { xIdx: 0, yIdx: Math.min(1, roles.shape.length - 1) };
        }
        // Prefer Energy × Photon (then kz); else Angle × Photon.
        let xIdx = roles.photon_axis;
        let yIdx = roles.energy_axis != null ? roles.energy_axis : roles.angle_axis;
        if (yIdx == null || yIdx === xIdx) {
            yIdx = xIdx === 0 ? 1 : 0;
        }
        return { xIdx, yIdx };
    }
    // Prefer angle on X and energy on Y for cuts; for maps prefer angle × beta.
    let xIdx = roles.angle_axis;
    let yIdx = roles.energy_axis != null ? roles.energy_axis : 0;
    if (roles.beta_axis != null && roles.energy_axis != null) {
        xIdx = roles.angle_axis;
        yIdx = roles.beta_axis;
    }
    if (xIdx === yIdx) {
        yIdx = xIdx === 0 ? 1 : 0;
    }
    return { xIdx, yIdx };
}

function processFixed(roles, xIdx, yIdx) {
    const fixed = {};
    const shape = roles.shape;
    const slider = el("ap-fixed-slider");
    for (let i = 0; i < shape.length; i++) {
        if (i === xIdx || i === yIdx) continue;
        if (!slider.disabled && processState.fixedAxis === i) {
            fixed[i] = Number(slider.value);
        } else {
            fixed[i] = Math.floor(shape[i] / 2);
        }
    }
    return fixed;
}

function convertPayload() {
    const roles = processState.roles;
    const { xIdx, yIdx } = processViewAxes(roles);
    processState.viewX = xIdx;
    processState.viewY = yIdx;
    return {
        angle_axis: roles.angle_axis,
        energy_axis: roles.energy_axis,
        beta_axis: roles.beta_axis,
        center: Number(el("ap-center").value),
        beta_center: Number(el("ap-beta-center").value),
        deg_per_unit: Number(el("ap-dpu").value),
        beta_deg_per_unit: Number(el("ap-beta-dpu").value),
        photon_energy: Number(el("ap-hv").value),
        work_function: Number(el("ap-phi").value),
        energy_mode: el("ap-emode").value,
        x_idx: xIdx,
        y_idx: yIdx,
        fixed: processFixed(roles, xIdx, yIdx),
        max_points: 512,
    };
}

function kzPayload() {
    const roles = processState.roles;
    const { xIdx, yIdx } = processViewAxes(roles);
    processState.viewX = xIdx;
    processState.viewY = yIdx;
    if (roles.photon_axis == null) {
        throw new Error("This dataset has no Photon Energy axis — load an hv scan.");
    }
    return {
        photon_axis: roles.photon_axis,
        work_function: Number(el("ap-kz-phi").value),
        inner_potential: Number(el("ap-vo").value),
        theta_deg: Number(el("ap-kz-theta").value),
        binding_ref: Number(el("ap-kz-eb").value),
        include_photon_momentum: el("ap-kz-q").checked,
        photon_incidence_angle: 45,
        x_idx: xIdx,
        y_idx: yIdx,
        fixed: processFixed(roles, xIdx, yIdx),
        max_points: 512,
    };
}

function scheduleProcessPreview() {
    clearTimeout(processState.debounce);
    processState.debounce = setTimeout(() => {
        updateProcessPreview().catch((err) => {
            el("ap-status").textContent = err.message;
        });
    }, 120);
}

async function loadProcessDataset(name) {
    if (!name) return;
    processEnsureViewers();
    syncProcessModeUI();
    const roles = await TensorSpecAPI.processRoles(name);
    processState.roles = roles;
    const photonNote =
        roles.photon_axis != null
            ? ` · photon ${roles.labels[roles.photon_axis]}`
            : "";
    el("ap-meta").textContent = `${roles.data_type} · ${roles.shape.join("×")} · ${roles.labels.join(", ")}${photonNote}`;
    el("ap-center").value = String(roles.center);
    el("ap-beta-center").value = String(roles.beta_center);
    el("ap-dpu").value = String(roles.deg_per_unit);
    el("ap-beta-dpu").value = String(roles.beta_deg_per_unit);
    el("ap-hv").value = String(roles.photon_energy);
    el("ap-phi").value = String(roles.work_function);
    el("ap-kz-phi").value = String(roles.work_function);
    el("ap-vo").value = String(roles.inner_potential ?? 15);
    el("ap-vo-slider").value = String(roles.inner_potential ?? 15);
    el("ap-store").value = processMode() === "kz" ? `${name}_kz` : `${name}_k`;
    el("ap-raw-badge").textContent = name;

    if (processMode() === "kz" && roles.photon_axis == null) {
        el("ap-status").textContent =
            "No Photon Energy axis on this dataset — pick an hv-dependent scan.";
    }

    const { xIdx, yIdx } = processViewAxes(roles);
    processState.viewX = xIdx;
    processState.viewY = yIdx;

    const slider = el("ap-fixed-slider");
    let fixedDim = null;
    for (let i = 0; i < roles.shape.length; i++) {
        if (i !== xIdx && i !== yIdx) {
            fixedDim = i;
            break;
        }
    }
    if (fixedDim == null) {
        slider.disabled = true;
        slider.min = 0;
        slider.max = 0;
        el("ap-fixed-label").textContent = "No extra axis";
        processState.fixedAxis = null;
    } else {
        slider.disabled = false;
        slider.min = 0;
        slider.max = roles.shape[fixedDim] - 1;
        slider.value = Math.floor(roles.shape[fixedDim] / 2);
        el("ap-fixed-label").textContent = `Fixed ${roles.labels[fixedDim]} index`;
        processState.fixedAxis = fixedDim;
    }

    const fixed = processFixed(roles, xIdx, yIdx);
    const raw = await TensorSpecAPI.tensorSlice(name, {
        x_idx: xIdx,
        y_idx: yIdx,
        fixed,
        max_points: 512,
    });
    processState.rawHeader = raw.header;
    processState.rawViewer.setData(raw.header, raw.values);
    if (processMode() === "inplane") {
        const centerXi = nearestIndex(raw.header.x_axis, Number(el("ap-center").value));
        processState.rawViewer.setCrosshair(centerXi, Math.floor(raw.header.shape[0] / 2));
        processState.rawViewer.setOverlays({
            vlines: xIdx === roles.angle_axis ? [Number(el("ap-center").value)] : [],
            hlines: yIdx === roles.angle_axis ? [Number(el("ap-center").value)] : [],
        });
    } else {
        processState.rawViewer.setOverlays({ polygons: [], vlines: [], hlines: [] });
    }
    await updateProcessPreview();
}

async function updateProcessPreview() {
    const name = el("ap-dataset").value;
    if (!name || !processState.roles) return;
    processEnsureViewers();

    if (processMode() === "kz") {
        const payload = kzPayload();
        const preview = await TensorSpecAPI.processKzPreview(name, payload);
        processState.kViewer.setData(preview.header, preview.values);
        el("ap-k-badge").textContent = `Vo ${Number(preview.header.inner_potential).toFixed(1)} eV · kz [${Number(
            preview.header.kz_min
        ).toFixed(2)}, ${Number(preview.header.kz_max).toFixed(2)}]`;

        const kOverlays = { polygons: [], vlines: [], hlines: [] };
        if (el("ap-perp-bz-toggle").checked && processState.perpBz) {
            const half = Number(processState.perpBz.half_g);
            const onX = preview.header.x_label === "kz";
            const onY = preview.header.y_label === "kz";
            const lo = onX ? preview.header.extent[0] : preview.header.extent[2];
            const hi = onX ? preview.header.extent[1] : preview.header.extent[3];
            const n0 = Math.floor(Math.min(lo, hi) / half) - 1;
            const n1 = Math.ceil(Math.max(lo, hi) / half) + 1;
            const lines = [];
            for (let n = n0; n <= n1; n++) lines.push(n * half);
            if (onX) kOverlays.vlines = lines;
            if (onY) kOverlays.hlines = lines;
        }
        processState.kViewer.setOverlays(kOverlays);
        processState.rawViewer.setOverlays({ polygons: [], vlines: [], hlines: [] });
        el("ap-status").textContent = `kz preview · Vo=${Number(el("ap-vo").value).toFixed(1)} eV`;
        return;
    }

    const payload = convertPayload();
    const preview = await TensorSpecAPI.processInplanePreview(name, payload);
    processState.kViewer.setData(preview.header, preview.values);
    el("ap-k-badge").textContent = `Eₖ ref ${Number(preview.header.e_kin_ref).toFixed(2)} eV`;

    const roles = processState.roles;
    const overlays = {
        vlines: [],
        hlines: [],
        polygons: [],
    };
    if (processState.viewX === roles.angle_axis) overlays.vlines.push(Number(el("ap-center").value));
    if (processState.viewY === roles.angle_axis) overlays.hlines.push(Number(el("ap-center").value));
    if (roles.beta_axis != null) {
        if (processState.viewX === roles.beta_axis) overlays.vlines.push(Number(el("ap-beta-center").value));
        if (processState.viewY === roles.beta_axis) overlays.hlines.push(Number(el("ap-beta-center").value));
    }
    processState.rawViewer.setOverlays(overlays);

    const kOverlays = { polygons: [], vlines: [0], hlines: roles.beta_axis != null ? [0] : [] };
    if (el("ap-bz-toggle").checked && processState.bzPolygon) {
        const poly = processState.bzPolygon;
        const pts = poly.kx.map((kx, i) => {
            const ky = poly.ky[i];
            let x = kx;
            let y = ky;
            if (preview.header.x_label === "ky") x = ky;
            if (preview.header.y_label === "kx") y = kx;
            if (preview.header.x_label === "kx") x = kx;
            if (preview.header.y_label === "ky") y = ky;
            if (preview.header.y_label === "Energy" || preview.header.x_label === "Energy") {
                const energyAxis = preview.header.y_label.includes("Energy")
                    ? preview.header.y_axis
                    : preview.header.x_axis;
                const midE = energyAxis[Math.floor(energyAxis.length / 2)];
                if (preview.header.x_label === "kx") return { x: kx, y: midE };
                if (preview.header.y_label === "kx") return { x: midE, y: kx };
            }
            return { x, y };
        });
        kOverlays.polygons = [pts];
    }
    processState.kViewer.setOverlays(kOverlays);
    el("ap-status").textContent = `Preview · mode ${preview.header.energy_mode || "auto"}`;
}

async function loadPerpBZ() {
    const crystal = el("ap-crystal").value;
    if (!crystal) {
        processState.perpBz = null;
        return;
    }
    processState.perpBz = await TensorSpecAPI.processPerpBZ({
        crystal_name: crystal,
        h: Number(el("ap-h").value),
        k: Number(el("ap-k").value),
        l: Number(el("ap-l").value),
        n_zones: 4,
    });
}

async function loadBZPolygon() {
    const crystal = el("ap-crystal").value;
    if (!crystal) {
        processState.bzPolygon = null;
        return;
    }
    processState.bzPolygon = await TensorSpecAPI.processSurfaceBZ({
        crystal_name: crystal,
        h: Number(el("ap-h").value),
        k: Number(el("ap-k").value),
        l: Number(el("ap-l").value),
    });
}

el("ap-refresh").addEventListener("click", () => {
    refreshProcessDatasets().catch((e) => {
        el("ap-status").textContent = e.message;
    });
});
el("ap-crystal-refresh").addEventListener("click", () => {
    refreshProcessCrystals().catch((e) => {
        el("ap-status").textContent = e.message;
    });
});
el("ap-dataset").addEventListener("change", () => {
    loadProcessDataset(el("ap-dataset").value).catch((e) => {
        el("ap-status").textContent = e.message;
    });
});
["ap-center", "ap-beta-center", "ap-hv", "ap-phi", "ap-dpu", "ap-beta-dpu", "ap-emode"].forEach((id) => {
    el(id).addEventListener("input", scheduleProcessPreview);
    el(id).addEventListener("change", scheduleProcessPreview);
});
el("ap-fixed-slider").addEventListener("input", async () => {
    const name = el("ap-dataset").value;
    if (!name || !processState.roles) return;
    try {
        const roles = processState.roles;
        const { xIdx, yIdx } = processViewAxes(roles);
        const fixed = processFixed(roles, xIdx, yIdx);
        const raw = await TensorSpecAPI.tensorSlice(name, {
            x_idx: xIdx,
            y_idx: yIdx,
            fixed,
            max_points: 512,
        });
        processState.rawHeader = raw.header;
        processState.rawViewer.setData(raw.header, raw.values);
        scheduleProcessPreview();
    } catch (err) {
        el("ap-status").textContent = err.message;
    }
});
el("ap-suggest").addEventListener("click", async () => {
    const name = el("ap-dataset").value;
    if (!name || !processState.roles) return;
    try {
        const roles = processState.roles;
        const { xIdx, yIdx } = processViewAxes(roles);
        const hint = await TensorSpecAPI.processSuggestCenter(name, {
            angle_axis: roles.angle_axis,
            energy_axis: roles.energy_axis,
            fixed: processFixed(roles, xIdx, yIdx),
        });
        el("ap-center").value = String(hint.value);
        el("ap-status").textContent = `Suggested center ${hint.value.toFixed(4)} (${hint.method}) — fine-tune by drag`;
        scheduleProcessPreview();
    } catch (err) {
        el("ap-status").textContent = err.message;
    }
});
el("ap-bz-toggle").addEventListener("change", async () => {
    try {
        if (el("ap-bz-toggle").checked) await loadBZPolygon();
        scheduleProcessPreview();
    } catch (err) {
        el("ap-status").textContent = err.message;
        el("ap-bz-toggle").checked = false;
    }
});
["ap-crystal", "ap-h", "ap-k", "ap-l"].forEach((id) => {
    el(id).addEventListener("change", async () => {
        try {
            if (processMode() === "kz" && el("ap-perp-bz-toggle").checked) {
                await loadPerpBZ();
            } else if (processMode() === "inplane" && el("ap-bz-toggle").checked) {
                await loadBZPolygon();
            }
            scheduleProcessPreview();
        } catch (err) {
            el("ap-status").textContent = err.message;
        }
    });
});
el("ap-apply").addEventListener("click", async () => {
    const name = el("ap-dataset").value;
    if (!name) return;
    el("ap-apply").disabled = true;
    try {
        if (processMode() === "kz") {
            const payload = {
                ...kzPayload(),
                store_as: el("ap-store").value,
                also_write_processed: true,
            };
            const result = await TensorSpecAPI.processKzApply(name, payload);
            el("ap-status").textContent = `Applied as ${result.name} (${result.shape.join("×")})${
                result.wrote_processed ? " · wrote /processed" : ""
            } · Vo=${Number(result.inner_potential).toFixed(1)}`;
        } else {
            const payload = {
                ...convertPayload(),
                store_as: el("ap-store").value,
                also_write_processed: true,
            };
            const result = await TensorSpecAPI.processInplaneApply(name, payload);
            el("ap-status").textContent = `Applied as ${result.name} (${result.shape.join("×")})${
                result.wrote_processed ? " · wrote /processed" : ""
            }`;
        }
        await refreshDatasets();
        await refreshProcessDatasets();
    } catch (err) {
        el("ap-status").textContent = err.message;
    } finally {
        el("ap-apply").disabled = false;
    }
});

el("ap-mode").addEventListener("change", () => {
    syncProcessModeUI();
    const name = el("ap-dataset").value;
    if (name) {
        el("ap-store").value = processMode() === "kz" ? `${name}_kz` : `${name}_k`;
        loadProcessDataset(name).catch((e) => {
            el("ap-status").textContent = e.message;
        });
    }
});

function syncVoFromSlider() {
    el("ap-vo").value = el("ap-vo-slider").value;
    scheduleProcessPreview();
}
function syncVoFromInput() {
    el("ap-vo-slider").value = el("ap-vo").value;
    scheduleProcessPreview();
}
el("ap-vo-slider").addEventListener("input", syncVoFromSlider);
el("ap-vo").addEventListener("input", syncVoFromInput);
["ap-kz-phi", "ap-kz-theta", "ap-kz-eb", "ap-kz-q"].forEach((id) => {
    el(id).addEventListener("input", scheduleProcessPreview);
    el(id).addEventListener("change", scheduleProcessPreview);
});

el("ap-perp-bz-toggle").addEventListener("change", async () => {
    try {
        if (el("ap-perp-bz-toggle").checked) await loadPerpBZ();
        scheduleProcessPreview();
    } catch (err) {
        el("ap-status").textContent = err.message;
        el("ap-perp-bz-toggle").checked = false;
    }
});

// Load process lists when the Process tab is selected
el("t3").addEventListener("change", () => {
    if (!el("t3").checked) return;
    processEnsureViewers();
    syncProcessModeUI();
    refreshProcessDatasets()
        .then(() => refreshProcessCrystals())
        .then(() => {
            if (el("ap-dataset").value) return loadProcessDataset(el("ap-dataset").value);
            return null;
        })
        .catch((e) => {
            el("ap-status").textContent = e.message;
        });
});
