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

function viewerUsesFrameNav(node = state.node) {
    return node === "raw"
        || (node === "processed" && !state.processedIsPaired)
        || isSeparatedNode(node);
}

function viewerFrameCount(node = state.node) {
    if (node === "raw") return state.nFrames;
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

    if (!state.hasProcessed && state.node === "processed") state.node = "raw";
    if (isSeparatedNode(state.node)) {
        const tag = state.node.slice("processed/".length);
        if (!state.separatedChannels.includes(tag)) {
            state.node = state.hasProcessed ? "processed" : "raw";
        }
    }
    state.frameIndex = Math.min(state.frameIndex, Math.max(0, state.nFrames - 1));
    state.pairIndex = Math.min(state.pairIndex, Math.max(0, state.nPairs - 1));
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
    resetRoi();
    setRoiMode(null);
    clearTimeout(state.frameTimer);
    dom.name.value = summary.name;
    configureViewer(summary);
    dom.vmin.disabled = false;
    dom.vmax.disabled = false;
    dom.status.textContent = `Loaded ${summary.name}: ${summary.n_frames} frame(s)`;
    dom.footerStatus.textContent = `${summary.name} · ${summary.n_frames} frame(s)`;
    showCsvState(summary);
    await showFrame(0);
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
