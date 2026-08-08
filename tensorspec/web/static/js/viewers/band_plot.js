/* Band structure plot: energy against distance along the k-path.
 *
 * Draws the payload from /api/dft/{name}/bands. Every eigenvalue is computed
 * by chinook through core.dft.band_service; this module only maps numbers to
 * pixels. Zero energy is the Fermi level, which is why it gets its own line.
 *
 * When `weights` are present (unfold_hex), bands are drawn faint and spectral
 * weight is shown as bright scatter points (ARPES-like intensity).
 * When `fat_weights` are present, orbital character uses the same scatter style;
 * if both exist, fat modulates unfold alpha.
 */

const MARGIN = { left: 58, right: 16, top: 16, bottom: 34 };
const WEIGHT_FLOOR = 0.05;

export class BandPlot {
    constructor(container) {
        this.container = container;

        this.canvas = document.createElement("canvas");
        this.canvas.style.width = "100%";
        this.canvas.style.height = "100%";
        this.canvas.style.display = "block";
        container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        this.result = null;
        this.range = { min: -6, max: 6 };

        this._resize = this._resize.bind(this);
        window.addEventListener("resize", this._resize);
        if (window.ResizeObserver) new ResizeObserver(this._resize).observe(container);
    }

    setResult(result) {
        this.result = result;
        this.draw();
    }

    setRange(min, max) {
        this.range = { min, max };
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
        if (!this.result) return;

        const rect = {
            left: MARGIN.left,
            top: MARGIN.top,
            width: width - MARGIN.left - MARGIN.right,
            height: height - MARGIN.top - MARGIN.bottom,
        };
        if (rect.width <= 0 || rect.height <= 0) return;

        const { k_dist, bands, node_positions, node_labels, weights, fat_weights } = this.result;
        const kMax = k_dist[k_dist.length - 1] || 1;
        const { min: eMin, max: eMax } = this.range;
        const eSpan = eMax - eMin || 1;
        const unfolded = Array.isArray(weights) && weights.length === bands.length;
        const fat = Array.isArray(fat_weights) && fat_weights.length === bands.length;
        const weighted = unfolded || fat;

        const px = (k) => rect.left + (k / kMax) * rect.width;
        const py = (e) => rect.top + rect.height - ((e - eMin) / eSpan) * rect.height;

        ctx.save();
        ctx.beginPath();
        ctx.rect(rect.left, rect.top, rect.width, rect.height);
        ctx.clip();

        ctx.strokeStyle = "#4b5563";
        ctx.lineWidth = 1;
        node_positions.forEach((k) => {
            ctx.beginPath();
            ctx.moveTo(px(k), rect.top);
            ctx.lineTo(px(k), rect.top + rect.height);
            ctx.stroke();
        });

        ctx.strokeStyle = "#f87171";
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(rect.left, py(0));
        ctx.lineTo(rect.left + rect.width, py(0));
        ctx.stroke();
        ctx.setLineDash([]);

        if (weighted) {
            ctx.strokeStyle = "rgba(96, 165, 250, 0.18)";
            ctx.lineWidth = 1;
            bands.forEach((band) => {
                ctx.beginPath();
                for (let i = 0; i < band.length; i++) {
                    const x = px(k_dist[i]);
                    const y = py(band[i]);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();
            });

            for (let b = 0; b < bands.length; b++) {
                const band = bands[b];
                for (let i = 0; i < band.length; i++) {
                    let w = 1;
                    if (unfolded) w *= (weights[b][i] ?? 0);
                    if (fat) w *= (fat_weights[b][i] ?? 0);
                    if (!unfolded && fat) w = fat_weights[b][i] ?? 0;
                    if (w < WEIGHT_FLOOR) continue;
                    const alpha = Math.min(1, 0.15 + 0.85 * w);
                    const r = 1.2 + 2.2 * w;
                    ctx.beginPath();
                    ctx.fillStyle = `rgba(251, 191, 36, ${alpha})`;
                    ctx.arc(px(k_dist[i]), py(band[i]), r, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        } else {
            ctx.strokeStyle = "#60a5fa";
            ctx.lineWidth = 1.2;
            bands.forEach((band) => {
                ctx.beginPath();
                for (let i = 0; i < band.length; i++) {
                    const x = px(k_dist[i]);
                    const y = py(band[i]);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();
            });
        }

        const overlay = this.result.overlay_bands;
        if (Array.isArray(overlay) && overlay.length) {
            ctx.save();
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = "#ef4444";
            ctx.lineWidth = 1.2;
            overlay.forEach((band) => {
                ctx.beginPath();
                for (let i = 0; i < band.length; i++) {
                    const x = px(k_dist[i]);
                    const y = py(band[i]);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();
            });
            ctx.restore();
        }

        ctx.restore();

        this._drawAxes(ctx, rect, eMin, eMax, node_positions, node_labels, px, {
            unfolded,
            fat,
        });
    }

    _drawAxes(ctx, rect, eMin, eMax, nodes, labels, px, flags = {}) {
        const { unfolded = false, fat = false } = flags;
        ctx.strokeStyle = "#3a3a4a";
        ctx.lineWidth = 1;
        ctx.strokeRect(rect.left, rect.top, rect.width, rect.height);

        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px ui-monospace, monospace";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";

        const ticks = 7;
        for (let i = 0; i < ticks; i++) {
            const fraction = i / (ticks - 1);
            const value = eMin + fraction * (eMax - eMin);
            ctx.fillText(value.toFixed(1), rect.left - 6, rect.top + rect.height - fraction * rect.height);
        }

        ctx.fillStyle = "#e2e8f0";
        ctx.font = "12px ui-sans-serif, system-ui";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        nodes.forEach((k, i) => {
            if (labels[i]) ctx.fillText(labels[i], px(k), rect.top + rect.height + 6);
        });

        ctx.fillStyle = "#cbd5e1";
        ctx.font = "11px ui-sans-serif, system-ui";
        ctx.save();
        ctx.translate(12, rect.top + rect.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = "top";
        ctx.fillText("Energy (eV)", 0, 0);
        ctx.restore();

        if (unfolded || fat) {
            ctx.fillStyle = "#fbbf24";
            ctx.font = "10px ui-sans-serif, system-ui";
            ctx.textAlign = "right";
            ctx.textBaseline = "top";
            let tag = "weight";
            if (unfolded && fat) tag = "unfold × orbital";
            else if (unfolded) tag = "spectral weight";
            else tag = "orbital weight";
            ctx.fillText(tag, rect.left + rect.width - 4, rect.top + 4);
        }
    }
}
