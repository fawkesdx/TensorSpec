const el = (id) => document.getElementById(id);

const dom = {
    name: el("peem-name"),
    tif: el("peem-tif"),
    zip: el("peem-zip"),
    serverPath: el("peem-server-path"),
    loadPath: el("peem-load-path"),
    status: el("peem-status"),
    csvControls: el("peem-csv-controls"),
    csvFile: el("peem-csv-file"),
    csvPath: el("peem-csv-path"),
    attachCsv: el("peem-attach-csv"),
    csvStatus: el("peem-csv-status"),
    csvPrompt: el("peem-csv-prompt"),
    csvCandidates: el("peem-csv-candidates"),
    continueWithout: el("peem-continue"),
    mode: el("peem-mode"),
    stackPairs: el("peem-stack-pairs"),
    separatePairs: el("peem-separate-pairs"),
    algo: el("peem-algo"),
    ref: el("peem-ref"),
    search: el("peem-search"),
    trackChannelRow: el("peem-track-channel-row"),
    trackChannel: el("peem-track-channel"),
    applyDrift: el("peem-apply-drift"),
    bgControls: el("peem-bg-controls"),
    bgUseRoi: el("peem-bg-use-roi"),
    bgE0: el("peem-bg-e0"),
    bgE1: el("peem-bg-e1"),
    bgEnsembleDelta: el("peem-bg-ensemble-delta"),
    bgEnsembleN: el("peem-bg-ensemble-n"),
    bgShowRaw: el("peem-bg-show-raw"),
    bgShowBg: el("peem-bg-show-bg"),
    bgShowBand: el("peem-bg-show-band"),
    bgShowSub: el("peem-bg-show-sub"),
    bgPlot: el("peem-bg-plot"),
    bgStatus: el("peem-bg-status"),
    bgPreview: el("peem-bg-preview"),
    bgApply: el("peem-bg-apply"),
    roiRect: el("peem-roi-rect"),
    roiEllipse: el("peem-roi-ellipse"),
    roiPolygon: el("peem-roi-polygon"),
    roiClose: el("peem-roi-close"),
    roiClear: el("peem-roi-clear"),
    roiStatus: el("peem-roi-status"),
    canvas: el("peem-canvas"),
    rawNode: el("peem-node-raw"),
    processedNode: el("peem-node-processed"),
    separatedNodes: el("peem-separated-nodes"),
    rawControls: el("peem-raw-controls"),
    processedControls: el("peem-processed-controls"),
    frame: el("peem-frame"),
    frameLabel: el("peem-frame-label"),
    pair: el("peem-pair"),
    pairLabel: el("peem-pair-label"),
    channel: el("peem-channel"),
    frameMeta: el("peem-frame-meta"),
    vmin: el("peem-vmin"),
    vmax: el("peem-vmax"),
    footerStatus: el("peem-footer-status"),
};

const state = {
    name: "",
    nFrames: 0,
    frameIndex: 0,
    node: "raw",
    hasProcessed: false,
    processedIsPaired: false,
    nProcessedFrames: 0,
    nPairs: 0,
    pairIndex: 0,
    channel: 0,
    channelTags: [],
    separatedChannels: [],
    separating: false,
    bgBusy: false,
    bgPreviewData: null,
    bgPreviewGen: 0,
    bgPreviewDirty: false,
    hasBackground: false,
    processedBgNode: null,
    nBgFrames: 0,
    energySource: null,
    bgDrag: null,
    bgPreviewTimer: null,
    vmin: 0,
    vmax: 1,
    frameData: null,
    requestId: 0,
    climCustomized: false,
    frameTimer: null,
    hasDrift: false,
    roi: null,
    roiMode: null,
    roiDraft: null,
    polygonPoints: [],
    isDrawing: false,
};

function isProcessedPaired(summary) {
    if (summary.processed_is_paired === true) return true;
    const nPairs = Number(summary.n_pairs);
    if (nPairs > 0) return true;
    const tags = summary.channel_tags || [];
    if (tags.length >= 2) return true;
    const shape = summary.processed_shape || [];
    return shape.length === 4;
}

function processedFrameCount(summary) {
    const fromMeta = Number(summary.n_processed_frames);
    if (fromMeta > 0) return fromMeta;
    const shape = summary.processed_shape || [];
    if (shape.length === 3) return Number(shape[0]) || 0;
    return 0;
}

function isSeparatedNode(node = state.node) {
    return typeof node === "string" && node.startsWith("processed/");
}

function isBgNode(node = state.node) {
    if (!isSeparatedNode(node)) return false;
    if (state.processedBgNode && node === state.processedBgNode) return true;
    const tag = node.slice("processed/".length);
    return tag === "bg" || tag.endsWith("_bg");
}

function viewerUsesFrameNav(node = state.node) {
    return node === "raw"
        || (node === "processed" && !state.processedIsPaired)
        || isSeparatedNode(node);
}

function viewerFrameCount(node = state.node) {
    if (node === "raw") return state.nFrames;
    if (isBgNode(node)) return state.nBgFrames > 0 ? state.nBgFrames : state.nFrames;
    if (isSeparatedNode(node)) return state.nPairs || state.nProcessedFrames;
    return state.processedIsPaired ? state.nPairs : state.nProcessedFrames;
}

function viewerFrameIndex(node = state.node) {
    if (node === "raw") return state.frameIndex;
    if (isSeparatedNode(node)) return state.pairIndex;
    return state.processedIsPaired ? state.pairIndex : state.frameIndex;
}

function processedNodeLabel() {
    return state.processedIsPaired ? "Paired" : "Processed";
}

function updateProcessedNodeLabel() {
    const label = dom.processedNode.closest("label");
    if (!label) return;
    for (const child of [...label.childNodes]) {
        if (child !== dom.processedNode) label.removeChild(child);
    }
    label.append(document.createTextNode(` ${processedNodeLabel()}`));
}

function onNodeRadioChange(node) {
    state.requestId += 1;
    clearTimeout(state.frameTimer);
    state.node = node;
    state.climCustomized = false;
    configureViewer({
        n_frames: state.nFrames,
        has_processed: state.hasProcessed,
        has_drift: state.hasDrift,
        processed_is_paired: state.processedIsPaired,
        n_processed_frames: state.nProcessedFrames,
        n_pairs: state.nPairs,
        channel_tags: state.channelTags,
        separated_channels: state.separatedChannels,
        has_background: state.hasBackground,
        has_processed_bg: state.nBgFrames > 0 || Boolean(state.processedBgNode),
        processed_bg_node: state.processedBgNode,
        energy_source: state.energySource,
    });
    showFrame(viewerFrameIndex());
}

function rebuildSeparatedNodeRadios() {
    dom.separatedNodes.replaceChildren();
    for (const tag of state.separatedChannels) {
        const label = document.createElement("label");
        label.className = "check";
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = "peem-node";
        radio.value = `processed/${tag}`;
        radio.checked = state.node === `processed/${tag}`;
        radio.addEventListener("change", () => {
            if (!radio.checked) return;
            onNodeRadioChange(radio.value);
        });
        label.append(radio, document.createTextNode(` ${tag}`));
        dom.separatedNodes.append(label);
    }
}

function setBusy(busy, message = "") {
    dom.tif.disabled = busy;
    dom.zip.disabled = busy;
    dom.serverPath.disabled = busy;
    dom.loadPath.disabled = busy;
    if (message) {
        dom.status.textContent = message;
        dom.footerStatus.textContent = message;
    }
}

function defaultBgWindow(energyOrCount) {
    if (Array.isArray(energyOrCount) && energyOrCount.length > 1) {
        const eMin = energyOrCount[0];
        const eMax = energyOrCount[energyOrCount.length - 1];
        const span = eMax - eMin;
        return { e0: eMin, e1: eMin + 0.2 * span };
    }
    const n = Math.max(2, Number(energyOrCount) || state.nFrames || 2);
    return { e0: 0, e1: 0.2 * (n - 1) };
}

function setDefaultBgWindow(energyOrCount) {
    const { e0, e1 } = defaultBgWindow(energyOrCount);
    dom.bgE0.value = String(e0);
    dom.bgE1.value = String(e1);
}

function buildBgPayload() {
    const deltaRaw = dom.bgEnsembleDelta.value.trim();
    const payload = {
        node: state.node,
        channel: state.channel,
        use_roi: dom.bgUseRoi.checked,
        e0: Number(dom.bgE0.value),
        e1: Number(dom.bgE1.value),
        ensemble_n: Math.max(1, Math.min(101, Number(dom.bgEnsembleN.value) || 21)),
    };
    if (deltaRaw) payload.ensemble_delta = Number(deltaRaw);
    if (payload.use_roi) payload.roi = state.roi;
    return payload;
}

function bgPlotLayout() {
    const width = dom.bgPlot.width;
    const height = dom.bgPlot.height;
    const pad = { left: 48, right: 12, top: 12, bottom: 28 };
    return {
        width,
        height,
        pad,
        plotW: width - pad.left - pad.right,
        plotH: height - pad.top - pad.bottom,
    };
}

function energyAtPlotX(xPx, energy) {
    const { pad, plotW } = bgPlotLayout();
    const frac = Math.max(0, Math.min(1, (xPx - pad.left) / plotW));
    const idx = frac * (energy.length - 1);
    const i0 = Math.floor(idx);
    const i1 = Math.min(energy.length - 1, i0 + 1);
    const t = idx - i0;
    return energy[i0] * (1 - t) + energy[i1] * t;
}

function plotXForEnergy(e, energy) {
    const { pad, plotW } = bgPlotLayout();
    let idx = 0;
    for (let i = 1; i < energy.length; i += 1) {
        if (energy[i] >= e) break;
        idx = i;
    }
    if (idx >= energy.length - 1) return pad.left + plotW;
    const e0 = energy[idx];
    const e1 = energy[idx + 1];
    const t = e1 === e0 ? 0 : (e - e0) / (e1 - e0);
    const frac = (idx + t) / (energy.length - 1);
    return pad.left + frac * plotW;
}

function drawBgPlot() {
    const ctx = dom.bgPlot.getContext("2d");
    const layout = bgPlotLayout();
    const { width, height, pad, plotW, plotH } = layout;
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, width, height);

    const data = state.bgPreviewData;
    if (!data?.energy?.length) {
        ctx.fillStyle = "#888";
        ctx.font = "12px sans-serif";
        ctx.fillText("Preview to show spectrum", pad.left, pad.top + plotH / 2);
        return;
    }

    const { energy, spectrum, bg, bg_std: bgStd, subtracted } = data;
    const series = [];
    if (dom.bgShowRaw.checked) series.push(...spectrum);
    if (dom.bgShowBg.checked) series.push(...bg);
    if (dom.bgShowBand.checked) {
        for (let i = 0; i < bg.length; i += 1) {
            series.push(bg[i] + bgStd[i], bg[i] - bgStd[i]);
        }
    }
    if (dom.bgShowSub.checked) series.push(...subtracted);

    let yMin = Infinity;
    let yMax = -Infinity;
    for (const v of series) {
        if (Number.isFinite(v)) {
            yMin = Math.min(yMin, v);
            yMax = Math.max(yMax, v);
        }
    }
    if (!Number.isFinite(yMin) || yMin === yMax) {
        yMin = 0;
        yMax = 1;
    }
    const yPad = 0.05 * (yMax - yMin || 1);
    yMin -= yPad;
    yMax += yPad;

    const xOf = (i) => pad.left + (i / (energy.length - 1)) * plotW;
    const yOf = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    ctx.strokeStyle = "#444";
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    if (dom.bgShowBand.checked && bgStd?.length) {
        ctx.fillStyle = "rgba(255, 140, 0, 0.2)";
        ctx.beginPath();
        for (let i = 0; i < energy.length; i += 1) {
            const x = xOf(i);
            const y = yOf(bg[i] + bgStd[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        for (let i = energy.length - 1; i >= 0; i -= 1) {
            ctx.lineTo(xOf(i), yOf(bg[i] - bgStd[i]));
        }
        ctx.closePath();
        ctx.fill();
    }

    const drawLine = (values, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < values.length; i += 1) {
            const v = values[i];
            if (!Number.isFinite(v)) continue;
            const x = xOf(i);
            const y = yOf(v);
            if (!started) {
                ctx.moveTo(x, y);
                started = true;
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    };

    if (dom.bgShowRaw.checked) drawLine(spectrum, "#ddd");
    if (dom.bgShowBg.checked) drawLine(bg, "#ff8c00");
    if (dom.bgShowSub.checked) drawLine(subtracted, "#4dd0e1");

    const e0 = Number(dom.bgE0.value);
    const e1 = Number(dom.bgE1.value);
    for (const [e, label] of [[e0, "e0"], [e1, "e1"]]) {
        if (!Number.isFinite(e)) continue;
        const x = plotXForEnergy(e, energy);
        ctx.strokeStyle = "#ffeb3b";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, pad.top + plotH);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#ffeb3b";
        ctx.font = "10px sans-serif";
        ctx.fillText(label, x + 2, pad.top + 10);
    }

    ctx.fillStyle = "rgba(255, 235, 59, 0.12)";
    const x0 = plotXForEnergy(Math.min(e0, e1), energy);
    const x1 = plotXForEnergy(Math.max(e0, e1), energy);
    ctx.fillRect(x0, pad.top, x1 - x0, plotH);

    ctx.fillStyle = "#888";
    ctx.font = "10px sans-serif";
    const xLabel = state.energySource === "csv" ? "Energy (eV)" : "Frame index";
    ctx.fillText(xLabel, pad.left, height - 8);
}

function nearestBgHandle(xPx, energy) {
    const e0 = Number(dom.bgE0.value);
    const e1 = Number(dom.bgE1.value);
    if (!Number.isFinite(e0) || !Number.isFinite(e1)) return null;
    const x0 = plotXForEnergy(e0, energy);
    const x1 = plotXForEnergy(e1, energy);
    const threshold = 8;
    if (Math.abs(xPx - x0) <= threshold) return "e0";
    if (Math.abs(xPx - x1) <= threshold) return "e1";
    const lo = Math.min(x0, x1);
    const hi = Math.max(x0, x1);
    if (xPx >= lo && xPx <= hi) return "window";
    return null;
}

function scheduleBgPreview() {
    if (state.bgBusy) {
        state.bgPreviewDirty = true;
        return;
    }
    clearTimeout(state.bgPreviewTimer);
    state.bgPreviewTimer = setTimeout(() => {
        state.bgPreviewTimer = null;
        if (state.name) previewBg();
    }, 300);
}

function updateBgControls(summary) {
    const enabled = (Number(summary?.n_frames) || state.nFrames) > 0;
    dom.bgControls.disabled = !enabled;
    dom.bgPreview.disabled = !enabled || state.bgBusy;
    dom.bgApply.disabled = !enabled || state.bgBusy;
    if (summary?.has_background !== undefined) {
        state.hasBackground = Boolean(summary.has_background);
    }
    if (summary?.processed_bg_node !== undefined) {
        state.processedBgNode = summary.processed_bg_node;
    }
    if (summary?.energy_source !== undefined) {
        state.energySource = summary.energy_source;
    }
}

async function loadStoredBgSpectrum(name) {
    if (!state.hasBackground) return;
    try {
        const data = await TensorSpecAPI.peemBgSpectrum(name);
        if (state.name !== name) return;
        state.bgPreviewData = data;
        dom.bgE0.value = String(data.e0);
        dom.bgE1.value = String(data.e1);
        state.energySource = data.energy_source;
        drawBgPlot();
        dom.bgStatus.textContent = `Stored background (${data.energy_source})`;
    } catch (_) {
        /* no stored analysis yet */
    }
}

async function previewBg() {
    if (!state.name) {
        dom.bgStatus.textContent = "Load a PEEM stack before preview.";
        return;
    }
    if (dom.bgUseRoi.checked && !state.roi) {
        dom.bgStatus.textContent = "Draw an ROI or uncheck Use ROI.";
        return;
    }

    const previewName = state.name;
    const gen = ++state.bgPreviewGen;
    state.bgBusy = true;
    dom.bgPreview.disabled = true;
    dom.bgApply.disabled = true;
    dom.bgStatus.textContent = "Previewing background…";
    try {
        const payload = buildBgPayload();
        const data = await TensorSpecAPI.peemBgPreview(previewName, payload);
        if (state.name !== previewName || gen !== state.bgPreviewGen) return;
        state.bgPreviewData = data;
        dom.bgE0.value = String(data.e0);
        dom.bgE1.value = String(data.e1);
        state.energySource = data.energy_source;
        dom.bgStatus.textContent =
            `Fit: slope=${data.slope.toExponential(3)}, intercept=${data.intercept.toFixed(3)} · ${data.energy_source}`;
        drawBgPlot();
    } catch (err) {
        if (state.name !== previewName || gen !== state.bgPreviewGen) return;
        dom.bgStatus.textContent = String(err.message || err);
    } finally {
        state.bgBusy = false;
        dom.bgPreview.disabled = false;
        dom.bgApply.disabled = false;
        const dirty = state.bgPreviewDirty;
        state.bgPreviewDirty = false;
        if (dirty) scheduleBgPreview();
    }
}

async function applyBg() {
    if (!state.name) {
        dom.bgStatus.textContent = "Load a PEEM stack before apply.";
        return;
    }
    if (dom.bgUseRoi.checked && !state.roi) {
        dom.bgStatus.textContent = "Draw an ROI or uncheck Use ROI.";
        return;
    }

    const applyName = state.name;
    state.bgBusy = true;
    dom.bgPreview.disabled = true;
    dom.bgApply.disabled = true;
    setBusy(true, "Applying background…");
    dom.bgStatus.textContent = "Applying background to all frames…";
    try {
        const payload = buildBgPayload();
        const summary = await TensorSpecAPI.peemBgApply(applyName, payload);
        if (state.name !== applyName) return;
        const meta = await TensorSpecAPI.peemMeta(applyName);
        if (state.name !== applyName) return;
        state.nBgFrames = Number(summary.n_frames) || 0;
        state.hasBackground = Boolean(meta.has_background);
        state.processedBgNode = meta.processed_bg_node;
        state.energySource = meta.energy_source;
        configureViewer(meta);
        if (summary.processed_bg_node) {
            state.node = summary.processed_bg_node;
            configureViewer(meta);
        }
        dom.status.textContent =
            `Background applied → ${summary.processed_bg_node} (${summary.n_frames} frames)`;
        dom.footerStatus.textContent = `${state.name} · background applied`;
        dom.bgStatus.textContent = `Applied · ${summary.energy_source} · view ${summary.processed_bg_node}`;
        await loadStoredBgSpectrum(applyName);
        await showFrame(viewerFrameIndex(), applyName);
    } catch (err) {
        if (state.name !== applyName) return;
        dom.bgStatus.textContent = String(err.message || err);
        dom.footerStatus.textContent = "Background apply failed";
    } finally {
        state.bgBusy = false;
        dom.bgPreview.disabled = false;
        dom.bgApply.disabled = false;
        setBusy(false);
        const dirty = state.bgPreviewDirty;
        state.bgPreviewDirty = false;
        if (dirty) scheduleBgPreview();
    }
}

function updateDriftControls() {
    const showTrack = state.hasProcessed && state.processedIsPaired;
    dom.trackChannelRow.hidden = !showTrack;
    dom.roiClose.hidden = state.roiMode !== "polygon";
}

function resetRoi() {
    state.roi = null;
    state.roiDraft = null;
    state.polygonPoints = [];
    state.isDrawing = false;
    updateRoiStatus();
    renderFrame();
}

function updateRoiStatus() {
    if (state.roiMode === "polygon" && state.polygonPoints.length) {
        dom.roiStatus.textContent = `Polygon: ${state.polygonPoints.length} point(s)`;
        return;
    }
    if (state.roi) {
        dom.roiStatus.textContent = `${state.roi.kind} ROI set`;
        return;
    }
    dom.roiStatus.textContent = state.roiMode
        ? `Draw ${state.roiMode} on canvas`
        : "No ROI";
}

function setRoiMode(mode) {
    state.roiMode = mode;
    state.roiDraft = null;
    state.polygonPoints = [];
    state.isDrawing = false;
    for (const [key, button] of [
        ["rect", dom.roiRect],
        ["ellipse", dom.roiEllipse],
        ["polygon", dom.roiPolygon],
    ]) {
        button.classList.toggle("btn--primary", mode === key);
    }
    dom.roiClose.hidden = mode !== "polygon";
    updateRoiStatus();
    renderFrame();
}

function canvasPixelFromEvent(event) {
    const rect = dom.canvas.getBoundingClientRect();
    const scaleX = dom.canvas.width / rect.width;
    const scaleY = dom.canvas.height / rect.height;
    const x = Math.floor((event.clientX - rect.left) * scaleX);
    const y = Math.floor((event.clientY - rect.top) * scaleY);
    return {
        x: Math.max(0, Math.min(dom.canvas.width - 1, x)),
        y: Math.max(0, Math.min(dom.canvas.height - 1, y)),
    };
}

function drawRoiOverlay(context) {
    context.save();
    context.strokeStyle = "#00ff88";
    context.fillStyle = "rgba(0, 255, 136, 0.12)";
    context.lineWidth = 1;

    const drawRect = (x0, y0, x1, y1, fill = false) => {
        const left = Math.min(x0, x1);
        const top = Math.min(y0, y1);
        const width = Math.abs(x1 - x0);
        const height = Math.abs(y1 - y0);
        if (width < 1 || height < 1) return;
        if (fill) context.fillRect(left + 0.5, top + 0.5, width, height);
        context.strokeRect(left + 0.5, top + 0.5, width, height);
    };

    const drawEllipse = (cx, cy, rx, ry, fill = false) => {
        if (rx < 0.5 || ry < 0.5) return;
        context.beginPath();
        context.ellipse(cx + 0.5, cy + 0.5, rx, ry, 0, 0, Math.PI * 2);
        if (fill) context.fill();
        context.stroke();
    };

    const drawPolygon = (points, closed = false) => {
        if (points.length < 2) return;
        context.beginPath();
        context.moveTo(points[0][0] + 0.5, points[0][1] + 0.5);
        for (let i = 1; i < points.length; i += 1) {
            context.lineTo(points[i][0] + 0.5, points[i][1] + 0.5);
        }
        if (closed && points.length >= 3) {
            context.closePath();
            context.fill();
        }
        context.stroke();
    };

    const target = state.roiDraft || state.roi;
    if (target) {
        if (target.kind === "rect") {
            drawRect(target.x0, target.y0, target.x1, target.y1, !state.roiDraft);
        } else if (target.kind === "ellipse") {
            if (state.roiDraft) {
                const cx = (target.x0 + target.x1) / 2;
                const cy = (target.y0 + target.y1) / 2;
                const rx = Math.abs(target.x1 - target.x0) / 2;
                const ry = Math.abs(target.y1 - target.y0) / 2;
                drawEllipse(cx, cy, rx, ry, false);
            } else {
                drawEllipse(target.cx, target.cy, target.rx, target.ry, true);
            }
        } else if (target.kind === "polygon" && target.points?.length) {
            drawPolygon(target.points, true);
        }
    } else if (state.roiMode === "polygon" && state.polygonPoints.length) {
        drawPolygon(state.polygonPoints, false);
    }

    context.restore();
}

function finalizeRectDraft(draft) {
    const x0 = Math.min(draft.x0, draft.x1);
    const y0 = Math.min(draft.y0, draft.y1);
    const x1 = Math.max(draft.x0, draft.x1);
    const y1 = Math.max(draft.y0, draft.y1);
    if (x1 - x0 < 2 || y1 - y0 < 2) return null;
    return { kind: "rect", x0, y0, x1, y1 };
}

function finalizeEllipseDraft(draft) {
    const cx = (draft.x0 + draft.x1) / 2;
    const cy = (draft.y0 + draft.y1) / 2;
    const rx = Math.abs(draft.x1 - draft.x0) / 2;
    const ry = Math.abs(draft.y1 - draft.y0) / 2;
    if (rx < 1 || ry < 1) return null;
    return { kind: "ellipse", cx, cy, rx, ry };
}

function closePolygon() {
    if (state.polygonPoints.length < 3) {
        dom.roiStatus.textContent = "Polygon needs at least 3 points";
        return;
    }
    state.roi = {
        kind: "polygon",
        points: state.polygonPoints.map(([x, y]) => [x, y]),
    };
    state.polygonPoints = [];
    updateRoiStatus();
    renderFrame();
}

function configureViewer(summary) {
    state.nFrames = Number(summary.n_frames) || 0;
    state.hasProcessed = Boolean(summary.has_processed);
    state.hasDrift = Boolean(summary.has_drift);
    state.processedIsPaired = state.hasProcessed && isProcessedPaired(summary);
    state.nProcessedFrames = state.hasProcessed ? processedFrameCount(summary) : 0;
    state.nPairs = state.processedIsPaired ? Number(summary.n_pairs) || 0 : 0;
    state.channelTags = state.processedIsPaired ? (summary.channel_tags || []) : [];
    state.separatedChannels = summary.separated_channels || [];

    if (summary.has_processed_bg === true) {
        if (Number(summary.n_bg_frames) > 0) {
            state.nBgFrames = Number(summary.n_bg_frames);
        } else if (!state.nBgFrames) {
            state.nBgFrames = state.nFrames;
        }
    } else if (summary.has_processed_bg === false) {
        state.nBgFrames = 0;
    }

    if (!state.hasProcessed && state.node === "processed") state.node = "raw";
    if (isSeparatedNode(state.node)) {
        const tag = state.node.slice("processed/".length);
        if (!state.separatedChannels.includes(tag)) {
            state.node = state.hasProcessed ? "processed" : "raw";
        }
    }
    state.frameIndex = Math.min(state.frameIndex, Math.max(0, state.nFrames - 1));
    state.pairIndex = Math.min(state.pairIndex, Math.max(0, state.nPairs - 1));
    if (isBgNode(state.node)) {
        const bgCount = viewerFrameCount(state.node);
        state.pairIndex = Math.min(state.pairIndex, Math.max(0, bgCount - 1));
    } else if (isSeparatedNode(state.node)) {
        const sepCount = state.nPairs || state.nProcessedFrames;
        state.pairIndex = Math.min(state.pairIndex, Math.max(0, sepCount - 1));
    }
    if (state.node === "processed" && !state.processedIsPaired) {
        state.frameIndex = Math.min(
            state.frameIndex,
            Math.max(0, state.nProcessedFrames - 1),
        );
    }
    state.channel = Math.min(state.channel, Math.max(0, state.channelTags.length - 1));

    const frameCount = viewerFrameCount();
    const frameIndex = viewerFrameIndex();
    const useFrameNav = viewerUsesFrameNav();
    const usePairNav = state.node === "processed" && state.processedIsPaired;

    dom.processedNode.disabled = !state.hasProcessed;
    dom.separatePairs.disabled = state.separating || !state.processedIsPaired;
    dom.rawNode.checked = state.node === "raw";
    dom.processedNode.checked = state.node === "processed";
    updateProcessedNodeLabel();
    rebuildSeparatedNodeRadios();
    dom.rawControls.hidden = !useFrameNav;
    dom.processedControls.hidden = !usePairNav;

    dom.frame.min = "0";
    dom.frame.max = String(Math.max(0, frameCount - 1));
    dom.frame.value = String(frameIndex);
    dom.frame.disabled = frameCount < 2;
    dom.frameLabel.textContent = frameCount
        ? `${frameIndex + 1} / ${frameCount}`
        : "0 / 0";

    dom.pair.min = "0";
    dom.pair.max = String(Math.max(0, state.nPairs - 1));
    dom.pair.value = String(state.pairIndex);
    dom.pair.disabled = state.nPairs < 2;
    dom.pairLabel.textContent = state.nPairs
        ? `${state.pairIndex + 1} / ${state.nPairs}`
        : "0 / 0";

    dom.channel.replaceChildren();
    state.channelTags.forEach((tag, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = tag;
        dom.channel.append(option);
    });
    dom.channel.value = String(state.channel);
    dom.channel.disabled = !usePairNav || state.channelTags.length < 2;
    updateDriftControls();
    updateBgControls(summary);
}

function renderFrame() {
    if (!state.frameData) return;
    const { shape, intensity } = state.frameData;
    const [height, width] = shape;
    dom.canvas.width = width;
    dom.canvas.height = height;

    const context = dom.canvas.getContext("2d");
    const image = context.createImageData(width, height);
    const low = Math.min(state.vmin, state.vmax);
    const high = Math.max(state.vmin, state.vmax);
    const span = high - low;

    for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
            const value = intensity[y][x];
            const scaled = Number.isFinite(value) && span > 0
                ? Math.round(255 * Math.max(0, Math.min(1, (value - low) / span)))
                : 0;
            const offset = 4 * (y * width + x);
            image.data[offset] = scaled;
            image.data[offset + 1] = scaled;
            image.data[offset + 2] = scaled;
            image.data[offset + 3] = 255;
        }
    }
    context.putImageData(image, 0, 0);
    drawRoiOverlay(context);
}

async function showFrame(index, expectedName = state.name) {
    if (!expectedName || state.name !== expectedName) return;
    const requestId = ++state.requestId;
    const count = viewerFrameCount();
    if (!count) return;
    const clamped = Math.max(0, Math.min(count - 1, Number(index) || 0));
    const usePairNav = state.node === "processed" && state.processedIsPaired;
    const useSeparatedNav = isSeparatedNode(state.node);
    const itemName = usePairNav ? "pair" : "frame";
    dom.frameMeta.textContent = `Loading ${itemName} ${clamped + 1}…`;
    try {
        const frame = await TensorSpecAPI.peemFrame(expectedName, clamped, {
            node: state.node,
            channel: state.channel,
        });
        if (requestId !== state.requestId || state.name !== expectedName) return;
        if (usePairNav) {
            state.pairIndex = clamped;
            dom.pair.value = String(clamped);
            dom.pairLabel.textContent = `${clamped + 1} / ${state.nPairs}`;
        } else if (useSeparatedNav) {
            state.pairIndex = clamped;
            dom.frame.value = String(clamped);
            dom.frameLabel.textContent = `${clamped + 1} / ${count}`;
        } else {
            state.frameIndex = clamped;
            dom.frame.value = String(clamped);
            dom.frameLabel.textContent = `${clamped + 1} / ${count}`;
        }
        state.frameData = frame;
        if (!state.climCustomized) {
            state.vmin = frame.vmin;
            state.vmax = frame.vmax;
            dom.vmin.value = String(frame.vmin);
            dom.vmax.value = String(frame.vmax);
        }
        const details = usePairNav
            ? [
                `Pair ${clamped + 1}`,
                frame.channel_tag,
                `${frame.shape[1]} × ${frame.shape[0]}`,
            ]
            : useSeparatedNav
                ? [
                    state.node.slice("processed/".length),
                    `Frame ${clamped + 1}`,
                    `${frame.shape[1]} × ${frame.shape[0]}`,
                ]
                : [frame.frame_name, frame.pol, `${frame.shape[1]} × ${frame.shape[0]}`];
        dom.frameMeta.textContent = details.filter(Boolean).join(" · ");
        renderFrame();
    } catch (error) {
        if (requestId === state.requestId && state.name === expectedName) {
            dom.frameMeta.textContent = `Frame error: ${error.message}`;
        }
    }
}

function showCsvState(summary) {
    dom.csvControls.disabled = false;
    dom.csvPrompt.hidden = Boolean(summary.csv_attached);
    dom.csvCandidates.replaceChildren();

    if (summary.csv_attached) {
        dom.csvStatus.textContent = summary.I0_present
            ? "CSV + I0 OK"
            : "CSV attached; I0 not found";
        return;
    }

    dom.csvStatus.textContent = "No beamline CSV attached";
    for (const path of summary.csv_candidates || []) {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "btn";
        button.textContent = path;
        button.addEventListener("click", () => {
            dom.csvPath.value = path;
            attachCsv();
        });
        item.append(button);
        dom.csvCandidates.append(item);
    }
}

async function acceptLoad(summary) {
    state.name = summary.name;
    state.frameIndex = 0;
    state.node = "raw";
    state.hasProcessed = false;
    state.processedIsPaired = false;
    state.nProcessedFrames = 0;
    state.nPairs = 0;
    state.pairIndex = 0;
    state.channel = 0;
    state.channelTags = [];
    state.separatedChannels = [];
    state.frameData = null;
    state.climCustomized = false;
    state.hasDrift = false;
    state.bgBusy = false;
    state.bgPreviewGen = 0;
    state.bgPreviewDirty = false;
    clearTimeout(state.bgPreviewTimer);
    state.bgPreviewTimer = null;
    state.bgPreviewData = null;
    state.hasBackground = false;
    state.processedBgNode = null;
    state.nBgFrames = 0;
    state.energySource = null;
    setDefaultBgWindow(summary.n_frames);
    resetRoi();
    setRoiMode(null);
    clearTimeout(state.frameTimer);
    dom.name.value = summary.name;
    if (summary.has_processed_bg) {
        state.nBgFrames = Number(summary.n_frames) || 0;
    }
    configureViewer(summary);
    dom.vmin.disabled = false;
    dom.vmax.disabled = false;
    dom.status.textContent = `Loaded ${summary.name}: ${summary.n_frames} frame(s)`;
    dom.footerStatus.textContent = `${summary.name} · ${summary.n_frames} frame(s)`;
    showCsvState(summary);
    updateBgControls(summary);
    await showFrame(0);
    await loadStoredBgSpectrum(summary.name);
}

async function separatePairs() {
    if (!state.name) {
        dom.status.textContent = "Load a PEEM stack before separating.";
        return;
    }
    if (!state.processedIsPaired) return;

    const separateName = state.name;
    state.separating = true;
    dom.separatePairs.disabled = true;
    setBusy(true, "Separating channels…");
    try {
        const summary = await TensorSpecAPI.peemSeparate(separateName);
        if (state.name !== separateName) return;
        const meta = await TensorSpecAPI.peemMeta(separateName);
        if (state.name !== separateName) return;
        configureViewer({ ...meta, n_frames: meta.n_frames });
        dom.status.textContent =
            `Separated ${summary.channels.join(", ")} (${summary.n_frames} frames)`;
        dom.footerStatus.textContent =
            `${state.name} · separated ${summary.channels.join(", ")}`;
        if (summary.channels.length) {
            state.node = `processed/${summary.channels[0]}`;
            configureViewer(meta);
        }
        await showFrame(viewerFrameIndex(), separateName);
    } catch (err) {
        if (state.name !== separateName) return;
        dom.status.textContent = String(err.message || err);
        dom.footerStatus.textContent = "Separate failed";
    } finally {
        state.separating = false;
        setBusy(false);
        dom.separatePairs.disabled = !state.processedIsPaired;
    }
}

async function stackPairs() {
    if (!state.name) {
        dom.status.textContent = "Load a PEEM stack before pairing.";
        return;
    }

    const pairedName = state.name;
    dom.stackPairs.disabled = true;
    dom.mode.disabled = true;
    dom.status.textContent = "Stacking contrast pairs…";
    dom.footerStatus.textContent = "Stacking contrast pairs…";
    try {
        const result = await TensorSpecAPI.peemPair(pairedName, dom.mode.value);
        if (state.name !== pairedName) return;
        const summary = await TensorSpecAPI.peemMeta(pairedName);
        if (state.name !== pairedName) return;
        state.node = "processed";
        state.frameIndex = 0;
        state.pairIndex = 0;
        state.channel = 0;
        state.separatedChannels = [];
        state.frameData = null;
        state.climCustomized = false;
        configureViewer(summary);
        const unpaired = result.unpaired_count === 1
            ? "1 unpaired frame"
            : `${result.unpaired_count} unpaired frames`;
        dom.status.textContent = `Stacked ${result.n_pairs} pair(s) · ${unpaired}`;
        dom.footerStatus.textContent = `${state.name} · ${result.n_pairs} pair(s)`;
        await showFrame(0, pairedName);
    } catch (error) {
        if (state.name !== pairedName) return;
        dom.status.textContent = `Pairing error: ${error.message}`;
        dom.footerStatus.textContent = "Pairing failed";
    } finally {
        dom.stackPairs.disabled = false;
        dom.mode.disabled = false;
    }
}

async function applyDrift() {
    if (!state.name) {
        dom.status.textContent = "Load a PEEM stack before drift correction.";
        return;
    }
    if (!state.roi) {
        dom.status.textContent = "Draw an ROI on the canvas first.";
        dom.roiStatus.textContent = "ROI required";
        return;
    }

    const driftName = state.name;
    const source = state.hasProcessed ? "processed" : "raw";
    const refIndex = Math.max(0, Number(dom.ref.value) || 0);
    const searchRadius = Math.max(1, Math.min(200, Number(dom.search.value) || 20));
    const trackChannel = Number(dom.trackChannel.value) || 0;
    const payload = {
        source,
        ref_index: refIndex,
        search_radius: searchRadius,
        track_channel: trackChannel,
        roi: state.roi,
    };

    dom.applyDrift.disabled = true;
    dom.ref.disabled = true;
    dom.search.disabled = true;
    dom.trackChannel.disabled = true;
    dom.status.textContent = "Applying drift correction…";
    dom.footerStatus.textContent = "Applying drift correction…";
    try {
        const result = await TensorSpecAPI.peemDrift(driftName, payload);
        if (state.name !== driftName) return;
        const summary = await TensorSpecAPI.peemMeta(driftName);
        if (state.name !== driftName) return;
        state.node = "processed";
        state.frameIndex = 0;
        state.pairIndex = 0;
        state.channel = 0;
        state.separatedChannels = [];
        state.frameData = null;
        state.climCustomized = false;
        state.hasDrift = true;
        configureViewer(summary);
        dom.status.textContent =
            `Drift applied · max |dx|=${result.max_abs_dx} · max |dy|=${result.max_abs_dy}`;
        dom.footerStatus.textContent = `${state.name} · drift corrected`;
        await showFrame(0, driftName);
    } catch (error) {
        if (state.name !== driftName) return;
        dom.status.textContent = `Drift error: ${error.message}`;
        dom.footerStatus.textContent = "Drift correction failed";
    } finally {
        if (state.name === driftName) {
            dom.applyDrift.disabled = false;
            dom.ref.disabled = false;
            dom.search.disabled = false;
            dom.trackChannel.disabled = false;
        }
    }
}

async function loadPeem(source) {
    const name = dom.name.value.trim();
    setBusy(true, "Loading PEEM data…");
    try {
        const summary = await TensorSpecAPI.loadPeem({ ...source, name });
        await acceptLoad(summary);
    } catch (error) {
        dom.status.textContent = `Load error: ${error.message}`;
        dom.footerStatus.textContent = "Load failed";
    } finally {
        setBusy(false);
    }
}

async function attachCsv() {
    if (!state.name) return;
    const csvFile = dom.csvFile.files[0] || null;
    const csvPath = dom.csvPath.value.trim();
    if (!csvFile && !csvPath) {
        dom.csvStatus.textContent = "Choose a CSV file or enter a server path.";
        return;
    }

    dom.attachCsv.disabled = true;
    dom.csvStatus.textContent = "Attaching CSV…";
    try {
        const summary = await TensorSpecAPI.peemAttachCsv(state.name, { csvFile, csvPath });
        showCsvState(summary);
    } catch (error) {
        dom.csvStatus.textContent = `CSV error: ${error.message}`;
    } finally {
        dom.attachCsv.disabled = false;
    }
}

dom.tif.addEventListener("change", () => {
    const file = dom.tif.files[0];
    if (file) loadPeem({ file });
});

dom.zip.addEventListener("change", () => {
    const file = dom.zip.files[0];
    if (file) loadPeem({ file });
});

dom.loadPath.addEventListener("click", () => {
    const serverPath = dom.serverPath.value.trim();
    if (!serverPath) {
        dom.status.textContent = "Enter a server TIF or folder path.";
        return;
    }
    loadPeem({ serverPath });
});

dom.serverPath.addEventListener("keydown", (event) => {
    if (event.key === "Enter") dom.loadPath.click();
});

dom.attachCsv.addEventListener("click", attachCsv);
dom.stackPairs.addEventListener("click", stackPairs);
dom.separatePairs.addEventListener("click", separatePairs);
dom.applyDrift.addEventListener("click", applyDrift);
dom.bgPreview.addEventListener("click", previewBg);
dom.bgApply.addEventListener("click", applyBg);
for (const input of [dom.bgShowRaw, dom.bgShowBg, dom.bgShowBand, dom.bgShowSub]) {
    input.addEventListener("change", drawBgPlot);
}
for (const input of [dom.bgE0, dom.bgE1]) {
    input.addEventListener("change", () => {
        drawBgPlot();
        scheduleBgPreview();
    });
}
dom.bgPlot.addEventListener("mousedown", (event) => {
    const data = state.bgPreviewData;
    if (!data?.energy?.length) return;
    event.preventDefault();
    const rect = dom.bgPlot.getBoundingClientRect();
    const scaleX = dom.bgPlot.width / rect.width;
    const xPx = (event.clientX - rect.left) * scaleX;
    const handle = nearestBgHandle(xPx, data.energy);
    if (handle) {
        state.bgDrag = { handle, energy: data.energy, startE0: Number(dom.bgE0.value), startE1: Number(dom.bgE1.value), startX: xPx };
        return;
    }
    const e = energyAtPlotX(xPx, data.energy);
    const e0 = Number(dom.bgE0.value);
    const e1 = Number(dom.bgE1.value);
    if (!Number.isFinite(e0) || !Number.isFinite(e1) || Math.abs(e - e0) <= Math.abs(e - e1)) {
        dom.bgE0.value = String(e);
    } else {
        dom.bgE1.value = String(e);
    }
    drawBgPlot();
    scheduleBgPreview();
});
dom.bgPlot.addEventListener("mousemove", (event) => {
    if (!state.bgDrag) return;
    const rect = dom.bgPlot.getBoundingClientRect();
    const scaleX = dom.bgPlot.width / rect.width;
    const xPx = (event.clientX - rect.left) * scaleX;
    const e = energyAtPlotX(xPx, state.bgDrag.energy);
    const { handle, startE0, startE1, startX } = state.bgDrag;
    if (handle === "e0") {
        dom.bgE0.value = String(e);
    } else if (handle === "e1") {
        dom.bgE1.value = String(e);
    } else {
        const startE = energyAtPlotX(startX, state.bgDrag.energy);
        const delta = e - startE;
        dom.bgE0.value = String(startE0 + delta);
        dom.bgE1.value = String(startE1 + delta);
    }
    drawBgPlot();
});
dom.bgPlot.addEventListener("mouseup", () => {
    if (!state.bgDrag) return;
    state.bgDrag = null;
    scheduleBgPreview();
});
dom.bgPlot.addEventListener("mouseleave", () => {
    if (!state.bgDrag) return;
    state.bgDrag = null;
    scheduleBgPreview();
});
dom.roiRect.addEventListener("click", () => setRoiMode("rect"));
dom.roiEllipse.addEventListener("click", () => setRoiMode("ellipse"));
dom.roiPolygon.addEventListener("click", () => setRoiMode("polygon"));
dom.roiClose.addEventListener("click", closePolygon);
dom.roiClear.addEventListener("click", () => {
    resetRoi();
    setRoiMode(null);
});
dom.continueWithout.addEventListener("click", () => {
    dom.csvPrompt.hidden = true;
    dom.csvStatus.textContent = "Continuing without beamline CSV";
});

dom.frame.addEventListener("input", () => {
    const count = viewerFrameCount();
    const index = Math.max(0, Math.min(count - 1, Number(dom.frame.value) || 0));
    dom.frameLabel.textContent = `${index + 1} / ${count}`;
    state.requestId += 1;
    clearTimeout(state.frameTimer);
    state.frameTimer = setTimeout(() => showFrame(index), 125);
});

dom.pair.addEventListener("input", () => {
    const index = Math.max(0, Math.min(state.nPairs - 1, Number(dom.pair.value) || 0));
    dom.pairLabel.textContent = `${index + 1} / ${state.nPairs}`;
    state.requestId += 1;
    clearTimeout(state.frameTimer);
    state.frameTimer = setTimeout(() => showFrame(index), 125);
});

for (const input of [dom.rawNode, dom.processedNode]) {
    input.addEventListener("change", () => {
        if (!input.checked) return;
        onNodeRadioChange(input.value);
    });
}

dom.channel.addEventListener("change", () => {
    state.channel = Number(dom.channel.value) || 0;
    state.climCustomized = false;
    showFrame(state.pairIndex);
});

for (const input of [dom.vmin, dom.vmax]) {
    input.addEventListener("input", () => {
        const vmin = Number(dom.vmin.value);
        const vmax = Number(dom.vmax.value);
        if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) return;
        state.climCustomized = true;
        state.vmin = Math.min(vmin, vmax);
        state.vmax = Math.max(vmin, vmax);
        renderFrame();
    });
}

dom.canvas.addEventListener("mousedown", (event) => {
    if (!state.frameData || !state.roiMode) return;
    event.preventDefault();
    const { x, y } = canvasPixelFromEvent(event);
    if (state.roiMode === "polygon") {
        state.polygonPoints.push([x, y]);
        updateRoiStatus();
        renderFrame();
        return;
    }
    state.roiDraft = {
        kind: state.roiMode,
        x0: x,
        y0: y,
        x1: x,
        y1: y,
    };
    state.isDrawing = true;
    renderFrame();
});

dom.canvas.addEventListener("mousemove", (event) => {
    if (!state.isDrawing || !state.roiDraft) return;
    const { x, y } = canvasPixelFromEvent(event);
    state.roiDraft.x1 = x;
    state.roiDraft.y1 = y;
    renderFrame();
});

dom.canvas.addEventListener("mouseup", () => {
    if (!state.isDrawing || !state.roiDraft) return;
    const finalized = state.roiDraft.kind === "ellipse"
        ? finalizeEllipseDraft(state.roiDraft)
        : finalizeRectDraft(state.roiDraft);
    state.roiDraft = null;
    state.isDrawing = false;
    if (finalized) {
        state.roi = finalized;
    }
    updateRoiStatus();
    renderFrame();
});

dom.canvas.addEventListener("mouseleave", () => {
    if (!state.isDrawing) return;
    dom.canvas.dispatchEvent(new Event("mouseup"));
});

dom.canvas.addEventListener("dblclick", (event) => {
    if (state.roiMode !== "polygon") return;
    event.preventDefault();
    closePolygon();
});

drawBgPlot();
