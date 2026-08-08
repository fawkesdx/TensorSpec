/* 2D intensity map with an interactive crosshair.
 *
 * Draws the plane served by /api/arpes/{name}/slice. Every value and axis
 * comes from `core.tensor_ops`; this module maps intensities to colours and
 * turns pointer positions into sample indices.
 *
 * The server sends row 0 as the lowest Y coordinate, matching matplotlib's
 * origin='lower', while canvas image rows run top-down. The image is
 * therefore written bottom-up so physics and display agree.
 */
import { COLORMAPS } from "/static/js/viewers/colormaps.js";

const MARGIN = { left: 62, right: 12, top: 12, bottom: 40 };
const CROSSHAIR = "#22d3ee";

export class ImageViewer {
    constructor(container, { onCrosshair = null } = {}) {
        this.container = container;
        this.onCrosshair = onCrosshair;

        this.canvas = document.createElement("canvas");
        this.canvas.style.width = "100%";
        this.canvas.style.height = "100%";
        this.canvas.style.display = "block";
        this.canvas.style.cursor = "crosshair";
        container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        // Intensities are rasterised once per data change into an offscreen
        // buffer at data resolution, then scaled on every repaint.
        this.buffer = document.createElement("canvas");

        this.header = null;
        this.values = null;
        this.colormap = "magma";
        this.crosshair = { x: 0, y: 0 };
        this.window = { dx: 0, dy: 0 };
        this.dragging = false;
        this.overlays = { polygons: [], vlines: [], hlines: [], polylines: [] };
        this.simOverlay = null; // { values: Float32Array, vmin, vmax, alpha, colormap? }

        this.canvas.addEventListener("pointerdown", (e) => this._onPointer(e, true));
        this.canvas.addEventListener("pointermove", (e) => {
            if (this.dragging) this._onPointer(e, false);
        });
        window.addEventListener("pointerup", () => { this.dragging = false; });

        this._resize = this._resize.bind(this);
        window.addEventListener("resize", this._resize);
        if (window.ResizeObserver) new ResizeObserver(this._resize).observe(container);
    }

    setData(header, values) {
        this.header = header;
        this.values = values;
        this.crosshair = {
            x: Math.min(this.crosshair.x, header.shape[1] - 1),
            y: Math.min(this.crosshair.y, header.shape[0] - 1),
        };
        this._rasterize();
        this.draw();
    }

    setColormap(name) {
        this.colormap = name;
        this._rasterize();
        this.draw();
    }

    setWindow(dx, dy) {
        this.window = { dx, dy };
        this.draw();
    }

    setCrosshair(x, y) {
        this.crosshair = { x, y };
        this.draw();
    }

    setOverlays({ polygons = [], vlines = [], hlines = [], polylines = [] } = {}) {
        this.overlays = { polygons, vlines, hlines, polylines };
        this.draw();
    }

    setSimOverlay(sim) {
        this.simOverlay = sim;
        this.draw();
    }

    /* Maps intensities through the colour table into the offscreen buffer. */
    _rasterize() {
        if (!this.header) return;
        const [rows, cols] = this.header.shape;
        const table = COLORMAPS[this.colormap] || COLORMAPS.magma;
        const { vmin, vmax } = this.header;
        const span = vmax - vmin || 1;

        this.buffer.width = cols;
        this.buffer.height = rows;
        const bufferCtx = this.buffer.getContext("2d");
        const image = bufferCtx.createImageData(cols, rows);

        for (let row = 0; row < rows; row++) {
            // Flip: data row 0 is the lowest Y, image row 0 is the top.
            const imageRow = rows - 1 - row;
            for (let col = 0; col < cols; col++) {
                const level = (this.values[row * cols + col] - vmin) / span;
                const entry = Math.max(0, Math.min(255, Math.round(level * 255))) * 3;
                const offset = (imageRow * cols + col) * 4;
                image.data[offset] = table[entry];
                image.data[offset + 1] = table[entry + 1];
                image.data[offset + 2] = table[entry + 2];
                image.data[offset + 3] = 255;
            }
        }
        bufferCtx.putImageData(image, 0, 0);
    }

    _plotRect() {
        const ratio = window.devicePixelRatio || 1;
        return {
            left: MARGIN.left,
            top: MARGIN.top,
            width: this.canvas.width / ratio - MARGIN.left - MARGIN.right,
            height: this.canvas.height / ratio - MARGIN.top - MARGIN.bottom,
        };
    }

    /* Sample index -> pixel, using the left edge of each sample's cell. */
    _indexToPixel(xi, yi) {
        const [rows, cols] = this.header.shape;
        const rect = this._plotRect();
        return {
            x: rect.left + ((xi + 0.5) / cols) * rect.width,
            y: rect.top + rect.height - ((yi + 0.5) / rows) * rect.height,
        };
    }

    _onPointer(event, isStart) {
        if (!this.header) return;
        if (isStart) this.dragging = true;

        const bounds = this.canvas.getBoundingClientRect();
        const rect = this._plotRect();
        const [rows, cols] = this.header.shape;

        const fx = (event.clientX - bounds.left - rect.left) / rect.width;
        const fy = (event.clientY - bounds.top - rect.top) / rect.height;

        const xi = Math.round(fx * cols - 0.5);
        const yi = Math.round((1 - fy) * rows - 0.5);

        const x = Math.max(0, Math.min(cols - 1, xi));
        const y = Math.max(0, Math.min(rows - 1, yi));

        if (x === this.crosshair.x && y === this.crosshair.y) return;
        this.crosshair = { x, y };
        this.draw();
        if (this.onCrosshair) this.onCrosshair(x, y);
    }

    _resize() {
        const ratio = window.devicePixelRatio || 1;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if (!w || !h) return;
        this.canvas.width = w * ratio;
        this.canvas.height = h * ratio;
        this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        this.draw();
    }

    draw() {
        if (!this.canvas.width) { this._resize(); return; }
        const ratio = window.devicePixelRatio || 1;
        const ctx = this.ctx;
        const width = this.canvas.width / ratio;
        const height = this.canvas.height / ratio;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#12121a";
        ctx.fillRect(0, 0, width, height);
        if (!this.header) return;

        const rect = this._plotRect();
        if (rect.width <= 0 || rect.height <= 0) return;

        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(this.buffer, rect.left, rect.top, rect.width, rect.height);

        if (this.simOverlay && this.simOverlay.values && this.header) {
            this._drawSimOverlay(ctx, rect);
        }

        this._drawWindow(ctx, rect);
        this._drawOverlays(ctx, rect);
        this._drawCrosshair(ctx, rect);
        this._drawAxes(ctx, rect);
    }

    _drawSimOverlay(ctx, rect) {
        const [rows, cols] = this.header.shape;
        const sim = this.simOverlay;
        if (!sim.values || sim.values.length < rows * cols) return;
        const table = COLORMAPS[sim.colormap || "viridis"] || COLORMAPS.viridis || COLORMAPS.magma;
        const vmin = sim.vmin ?? 0;
        const vmax = sim.vmax ?? 1;
        const span = vmax - vmin || 1;
        const canvas = document.createElement("canvas");
        canvas.width = cols;
        canvas.height = rows;
        const bctx = canvas.getContext("2d");
        const image = bctx.createImageData(cols, rows);
        for (let row = 0; row < rows; row++) {
            const imageRow = rows - 1 - row;
            for (let col = 0; col < cols; col++) {
                const v = sim.values[row * cols + col];
                const level = (v - vmin) / span;
                const entry = Math.max(0, Math.min(255, Math.round(level * 255))) * 3;
                const offset = (imageRow * cols + col) * 4;
                image.data[offset] = table[entry];
                image.data[offset + 1] = table[entry + 1];
                image.data[offset + 2] = table[entry + 2];
                image.data[offset + 3] = v > vmin + 0.02 * span ? 255 : 0;
            }
        }
        bctx.putImageData(image, 0, 0);
        ctx.save();
        ctx.globalAlpha = sim.alpha ?? 0.45;
        ctx.drawImage(canvas, rect.left, rect.top, rect.width, rect.height);
        ctx.restore();
    }

    _dataToPixel(xData, yData, rect) {
        const extent = this.header.extent;
        const xSpan = extent[1] - extent[0] || 1;
        const ySpan = extent[3] - extent[2] || 1;
        return {
            x: rect.left + ((xData - extent[0]) / xSpan) * rect.width,
            y: rect.top + rect.height - ((yData - extent[2]) / ySpan) * rect.height,
        };
    }

    _drawOverlays(ctx, rect) {
        const { polygons, vlines, hlines, polylines } = this.overlays || {};
        ctx.save();
        if (polygons && polygons.length) {
            ctx.strokeStyle = "#22d3ee";
            ctx.lineWidth = 1.5;
            polygons.forEach((poly) => {
                if (!poly || poly.length < 2) return;
                ctx.beginPath();
                poly.forEach((pt, i) => {
                    const p = this._dataToPixel(pt.x, pt.y, rect);
                    if (i === 0) ctx.moveTo(p.x, p.y);
                    else ctx.lineTo(p.x, p.y);
                });
                ctx.stroke();
            });
        }
        if (polylines && polylines.length) {
            const palette = ["#f472b6", "#22d3ee", "#fbbf24", "#a78bfa", "#34d399", "#fb7185"];
            polylines.forEach((pl, idx) => {
                const pts = pl.points || pl;
                if (!pts || pts.length < 2) return;
                ctx.strokeStyle = pl.color || palette[idx % palette.length];
                ctx.lineWidth = pl.width || 1.4;
                ctx.setLineDash(pl.dash || []);
                ctx.beginPath();
                pts.forEach((pt, i) => {
                    const p = this._dataToPixel(pt.x, pt.y, rect);
                    if (i === 0) ctx.moveTo(p.x, p.y);
                    else ctx.lineTo(p.x, p.y);
                });
                ctx.stroke();
                ctx.setLineDash([]);
            });
        }
        ctx.strokeStyle = "#fbbf24";
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.25;
        (vlines || []).forEach((xData) => {
            const p = this._dataToPixel(xData, this.header.extent[2], rect);
            ctx.beginPath();
            ctx.moveTo(p.x, rect.top);
            ctx.lineTo(p.x, rect.top + rect.height);
            ctx.stroke();
        });
        (hlines || []).forEach((yData) => {
            const p = this._dataToPixel(this.header.extent[0], yData, rect);
            ctx.beginPath();
            ctx.moveTo(rect.left, p.y);
            ctx.lineTo(rect.left + rect.width, p.y);
            ctx.stroke();
        });
        ctx.setLineDash([]);
        ctx.restore();
    }

    _drawWindow(ctx, rect) {
        const [rows, cols] = this.header.shape;
        const { dx, dy } = this.window;
        if (dx === 0 && dy === 0) return;

        const x1 = Math.max(0, this.crosshair.x - dx);
        const x2 = Math.min(cols, this.crosshair.x + dx + 1);
        const y1 = Math.max(0, this.crosshair.y - dy);
        const y2 = Math.min(rows, this.crosshair.y + dy + 1);

        const left = rect.left + (x1 / cols) * rect.width;
        const right = rect.left + (x2 / cols) * rect.width;
        const bottom = rect.top + rect.height - (y1 / rows) * rect.height;
        const top = rect.top + rect.height - (y2 / rows) * rect.height;

        ctx.fillStyle = "rgba(34, 211, 238, 0.2)";
        ctx.fillRect(left, top, right - left, bottom - top);
        ctx.strokeStyle = CROSSHAIR;
        ctx.setLineDash([2, 2]);
        ctx.lineWidth = 1.5;
        ctx.strokeRect(left, top, right - left, bottom - top);
        ctx.setLineDash([]);
    }

    _drawCrosshair(ctx, rect) {
        const point = this._indexToPixel(this.crosshair.x, this.crosshair.y);
        ctx.strokeStyle = CROSSHAIR;
        ctx.globalAlpha = 0.7;
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.moveTo(point.x, rect.top);
        ctx.lineTo(point.x, rect.top + rect.height);
        ctx.moveTo(rect.left, point.y);
        ctx.lineTo(rect.left + rect.width, point.y);
        ctx.stroke();

        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
    }

    _drawAxes(ctx, rect) {
        const { x_axis, y_axis, x_label, y_label, x_unit, y_unit } = this.header;

        ctx.strokeStyle = "#3a3a4a";
        ctx.lineWidth = 1;
        ctx.strokeRect(rect.left, rect.top, rect.width, rect.height);

        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px ui-monospace, monospace";

        const ticks = 5;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        for (let i = 0; i < ticks; i++) {
            const fraction = i / (ticks - 1);
            const value = x_axis[Math.round(fraction * (x_axis.length - 1))];
            ctx.fillText(value.toFixed(2), rect.left + fraction * rect.width, rect.top + rect.height + 6);
        }

        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (let i = 0; i < ticks; i++) {
            const fraction = i / (ticks - 1);
            const value = y_axis[Math.round(fraction * (y_axis.length - 1))];
            ctx.fillText(value.toFixed(2), rect.left - 6, rect.top + rect.height - fraction * rect.height);
        }

        ctx.fillStyle = "#cbd5e1";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        const xTitle = x_unit ? `${x_label} (${x_unit})` : x_label;
        ctx.fillText(xTitle, rect.left + rect.width / 2, rect.top + rect.height + MARGIN.bottom - 2);

        const yTitle = y_unit ? `${y_label} (${y_unit})` : y_label;
        ctx.save();
        ctx.translate(12, rect.top + rect.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = "top";
        ctx.fillText(yTitle, 0, 0);
        ctx.restore();
    }
}
