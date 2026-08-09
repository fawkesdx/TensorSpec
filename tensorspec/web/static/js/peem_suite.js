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
    canvas: el("peem-canvas"),
    frame: el("peem-frame"),
    frameLabel: el("peem-frame-label"),
    frameMeta: el("peem-frame-meta"),
    vmin: el("peem-vmin"),
    vmax: el("peem-vmax"),
    footerStatus: el("peem-footer-status"),
};

const state = {
    name: "",
    nFrames: 0,
    frameIndex: 0,
    vmin: 0,
    vmax: 1,
    frameData: null,
    requestId: 0,
    climCustomized: false,
    frameTimer: null,
};

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
}

async function showFrame(index) {
    if (!state.name) return;
    const requestId = ++state.requestId;
    const clamped = Math.max(0, Math.min(state.nFrames - 1, Number(index) || 0));
    dom.frameMeta.textContent = `Loading frame ${clamped + 1}…`;
    try {
        const frame = await TensorSpecAPI.peemFrame(state.name, clamped);
        if (requestId !== state.requestId) return;
        state.frameIndex = clamped;
        state.frameData = frame;
        dom.frame.value = String(clamped);
        dom.frameLabel.textContent = `${clamped + 1} / ${state.nFrames}`;
        if (!state.climCustomized) {
            state.vmin = frame.vmin;
            state.vmax = frame.vmax;
            dom.vmin.value = String(frame.vmin);
            dom.vmax.value = String(frame.vmax);
        }
        const details = [frame.frame_name, frame.pol, `${frame.shape[1]} × ${frame.shape[0]}`]
            .filter(Boolean);
        dom.frameMeta.textContent = details.join(" · ");
        renderFrame();
    } catch (error) {
        if (requestId === state.requestId) {
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
    state.nFrames = summary.n_frames;
    state.frameIndex = 0;
    state.frameData = null;
    state.climCustomized = false;
    clearTimeout(state.frameTimer);
    dom.name.value = summary.name;
    dom.frame.min = "0";
    dom.frame.max = String(Math.max(0, summary.n_frames - 1));
    dom.frame.value = "0";
    dom.frame.disabled = summary.n_frames < 2;
    dom.vmin.disabled = false;
    dom.vmax.disabled = false;
    dom.status.textContent = `Loaded ${summary.name}: ${summary.n_frames} frame(s)`;
    dom.footerStatus.textContent = `${summary.name} · ${summary.n_frames} frame(s)`;
    showCsvState(summary);
    await showFrame(0);
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
dom.continueWithout.addEventListener("click", () => {
    dom.csvPrompt.hidden = true;
    dom.csvStatus.textContent = "Continuing without beamline CSV";
});

dom.frame.addEventListener("input", () => {
    const index = Math.max(0, Math.min(state.nFrames - 1, Number(dom.frame.value) || 0));
    dom.frameLabel.textContent = `${index + 1} / ${state.nFrames}`;
    state.requestId += 1;
    clearTimeout(state.frameTimer);
    state.frameTimer = setTimeout(() => showFrame(index), 125);
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
