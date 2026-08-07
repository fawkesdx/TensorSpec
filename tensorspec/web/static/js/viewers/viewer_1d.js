/* 1D curve plot for the profiles served by /api/arpes/{name}/profiles.
 *
 * Supports a vertical orientation so an EDC can sit beside the image sharing
 * its energy axis, which is how the curve is read in practice: intensity runs
 * horizontally while the physical axis stays aligned with the map.
 */

const MARGIN = { left: 46, right: 10, top: 10, bottom: 28 };

export class LineViewer {
    constructor(container, { orientation = "horizontal", color = "#60a5fa" } = {}) {
        this.container = container;
        this.orientation = orientation;
        this.color = color;

        this.canvas = document.createElement("canvas");
        this.canvas.style.width = "100%";
        this.canvas.style.height = "100%";
        this.canvas.style.display = "block";
        container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        this.curve = null;
        this.marker = null;

        this._resize = this._resize.bind(this);
        window.addEventListener("resize", this._resize);
        if (window.ResizeObserver) new ResizeObserver(this._resize).observe(container);
    }

    setCurve(curve) {
        this.curve = curve;
        this.draw();
    }

    /* Position on the physical axis to mark, normally the crosshair. */
    setMarker(value) {
        this.marker = value;
        this.draw();
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
        if (!this.curve || !this.curve.values.length) return;

        const rect = {
            left: MARGIN.left,
            top: MARGIN.top,
            width: width - MARGIN.left - MARGIN.right,
            height: height - MARGIN.top - MARGIN.bottom,
        };
        if (rect.width <= 0 || rect.height <= 0) return;

        const { axis, values } = this.curve;
        const aMin = axis[0];
        const aMax = axis[axis.length - 1];
        const aSpan = aMax - aMin || 1;
        const vMin = Math.min(...values);
        const vMax = Math.max(...values);
        const vSpan = vMax - vMin || 1;

        ctx.strokeStyle = "#3a3a4a";
        ctx.lineWidth = 1;
        ctx.strokeRect(rect.left, rect.top, rect.width, rect.height);

        // Horizontal: physical axis across, intensity up.
        // Vertical: intensity across, physical axis up (aligned with the map).
        const toPixel = (a, v) => {
            const fa = (a - aMin) / aSpan;
            const fv = (v - vMin) / vSpan;
            return this.orientation === "horizontal"
                ? { x: rect.left + fa * rect.width, y: rect.top + rect.height - fv * rect.height }
                : { x: rect.left + fv * rect.width, y: rect.top + rect.height - fa * rect.height };
        };

        ctx.strokeStyle = this.color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < values.length; i++) {
            const p = toPixel(axis[i], values[i]);
            if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();

        if (this.marker !== null && this.marker >= Math.min(aMin, aMax) && this.marker <= Math.max(aMin, aMax)) {
            const fa = (this.marker - aMin) / aSpan;
            ctx.strokeStyle = "#22d3ee";
            ctx.globalAlpha = 0.6;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            if (this.orientation === "horizontal") {
                const x = rect.left + fa * rect.width;
                ctx.moveTo(x, rect.top);
                ctx.lineTo(x, rect.top + rect.height);
            } else {
                const y = rect.top + rect.height - fa * rect.height;
                ctx.moveTo(rect.left, y);
                ctx.lineTo(rect.left + rect.width, y);
            }
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
        }

        this._drawLabels(ctx, rect, aMin, aMax, vMin, vMax);
    }

    _drawLabels(ctx, rect, aMin, aMax, vMin, vMax) {
        const { label, unit } = this.curve;
        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px ui-monospace, monospace";

        const format = (v) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01)
            ? v.toExponential(1) : v.toFixed(2));

        if (this.orientation === "horizontal") {
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            ctx.fillText(format(aMin), rect.left, rect.top + rect.height + 4);
            ctx.fillText(format(aMax), rect.left + rect.width, rect.top + rect.height + 4);
            ctx.textAlign = "right";
            ctx.textBaseline = "top";
            ctx.fillText(format(vMax), rect.left - 4, rect.top);
            ctx.textBaseline = "bottom";
            ctx.fillText(format(vMin), rect.left - 4, rect.top + rect.height);
        } else {
            ctx.textAlign = "right";
            ctx.textBaseline = "bottom";
            ctx.fillText(format(aMin), rect.left - 4, rect.top + rect.height);
            ctx.textBaseline = "top";
            ctx.fillText(format(aMax), rect.left - 4, rect.top);
        }

        ctx.fillStyle = "#cbd5e1";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(unit ? `${label} (${unit})` : label, rect.left + 4, rect.top + 2);
    }
}
