/* Fetch wrapper for the TensorSpec FastAPI service.
   The only module that knows about URLs. No physics, no formatting. */

const TensorSpecAPI = (() => {
    async function request(path, options = {}) {
        const response = await fetch(path, {
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            ...options,
        });

        if (!response.ok) {
            let detail = `${response.status} ${response.statusText}`;
            try {
                const body = await response.json();
                if (body.detail) detail = body.detail;
            } catch (err) {
                /* response had no JSON body; keep the status text */
            }
            throw new Error(detail);
        }

        return response.json();
    }

    /* Multipart upload: the browser must set its own boundary header. */
    async function upload(path, formData) {
        const response = await fetch(path, {
            method: "POST",
            credentials: "same-origin",
            body: formData,
        });

        if (!response.ok) {
            let detail = `${response.status} ${response.statusText}`;
            try {
                const body = await response.json();
                if (body.detail) detail = body.detail;
            } catch (err) {
                /* response had no JSON body; keep the status text */
            }
            throw new Error(detail);
        }

        return response.json();
    }

    return {
        health: () => request("/api/health"),
        listItems: () => request("/api/workspace/items"),
        getItem: (name) => request(`/api/workspace/items/${encodeURIComponent(name)}`),
        seedDemo: (payload = {}) =>
            request("/api/workspace/demo", {
                method: "POST",
                body: JSON.stringify(payload),
            }),

        loadCif: (file, name = "") => {
            const form = new FormData();
            form.append("file", file);
            if (name) form.append("name", name);
            return upload("/api/crystal/load", form);
        },
        crystalSummary: (name) =>
            request(`/api/crystal/${encodeURIComponent(name)}/summary`),
        crystalGeometry: (name, payload = {}) =>
            request(`/api/crystal/${encodeURIComponent(name)}/geometry`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        crystalBZ: (name, payload = {}) =>
            request(`/api/crystal/${encodeURIComponent(name)}/bz`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        crystalTemplates: () => request("/api/crystal/templates"),
        crystalAddTemplate: (payload) =>
            request("/api/crystal/templates", {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        crystalStack: (payload) =>
            request("/api/crystal/stack", {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        crystalMoire: (payload) =>
            request("/api/crystal/moire", {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        crystalExfoliate: (payload) =>
            request("/api/crystal/exfoliate", {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        crystalMlipModels: () => request("/api/crystal/mlip/models"),
        crystalRelax: (name, payload) =>
            request(`/api/crystal/${encodeURIComponent(name)}/relax`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        crystalPush: (name, payload) =>
            request(`/api/crystal/${encodeURIComponent(name)}/push`, {
                method: "POST",
                body: JSON.stringify(
                    typeof payload === "string" ? { store_as: payload } : payload
                ),
            }),
        crystalCifUrl: (name) => `/api/crystal/${encodeURIComponent(name)}/cif`,
        /** GET unfiltered CIF, or POST with omit_atom_indices + nx/ny/nz/basis when omit non-empty. */
        crystalCifDownload: async (name, payload = {}) => {
            const omit = payload.omit_atom_indices || [];
            const url = `/api/crystal/${encodeURIComponent(name)}/cif`;
            const response = await fetch(url, {
                method: omit.length ? "POST" : "GET",
                credentials: "same-origin",
                headers: omit.length ? { "Content-Type": "application/json" } : undefined,
                body: omit.length ? JSON.stringify(payload) : undefined,
            });
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* ignore */
                }
                throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
            }
            return response.blob();
        },
        crystalCifPost: async (name, payload = {}) => {
            const response = await fetch(
                `/api/crystal/${encodeURIComponent(name)}/cif`,
                {
                    method: "POST",
                    credentials: "same-origin",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }
            );
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* ignore */
                }
                throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
            }
            return response.blob();
        },
        crystalExportScene: async (name, fmt, payload) => {
            const response = await fetch(
                `/api/crystal/${encodeURIComponent(name)}/export/${fmt}`,
                {
                    method: "POST",
                    credentials: "same-origin",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }
            );
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* ignore */
                }
                throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
            }
            return response.blob();
        },
        crystalGapPredict: (name, fidelity = "PBE") =>
            request(`/api/crystal/${encodeURIComponent(name)}/gap-predict`, {
                method: "POST",
                body: JSON.stringify({ fidelity }),
            }),

        loadArpes: (file, { name = "", logFile = null } = {}) => {
            const form = new FormData();
            form.append("file", file);
            if (name) form.append("name", name);
            if (logFile) form.append("log", logFile);
            return upload("/api/arpes/load", form);
        },
        processRoles: (name) =>
            request(`/api/arpes/process/${encodeURIComponent(name)}/roles`),
        processSuggestCenter: (name, payload) =>
            request(`/api/arpes/process/${encodeURIComponent(name)}/suggest-center`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        processInplanePreview: async (name, payload) => {
            const response = await fetch(
                `/api/arpes/process/${encodeURIComponent(name)}/inplane/preview`,
                {
                    method: "POST",
                    credentials: "same-origin",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }
            );
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* binary error */
                }
                throw new Error(detail);
            }
            const buffer = await response.arrayBuffer();
            const headerLength = new DataView(buffer).getUint32(0, true);
            const header = JSON.parse(
                new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength))
            );
            const values = new Float32Array(
                buffer,
                4 + headerLength,
                header.shape[0] * header.shape[1]
            );
            return { header, values };
        },
        processInplaneApply: (name, payload) =>
            request(`/api/arpes/process/${encodeURIComponent(name)}/inplane/apply`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        processSurfaceBZ: (payload) =>
            request("/api/arpes/process/surface-bz", {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        processKzPreview: async (name, payload) => {
            const response = await fetch(
                `/api/arpes/process/${encodeURIComponent(name)}/kz/preview`,
                {
                    method: "POST",
                    credentials: "same-origin",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }
            );
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* binary error */
                }
                throw new Error(detail);
            }
            const buffer = await response.arrayBuffer();
            const headerLength = new DataView(buffer).getUint32(0, true);
            const header = JSON.parse(
                new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength))
            );
            const values = new Float32Array(
                buffer,
                4 + headerLength,
                header.shape[0] * header.shape[1]
            );
            return { header, values };
        },
        processKzApply: (name, payload) =>
            request(`/api/arpes/process/${encodeURIComponent(name)}/kz/apply`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        processPerpBZ: (payload) =>
            request("/api/arpes/process/perp-bz", {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        tensorAxes: (name) =>
            request(`/api/arpes/${encodeURIComponent(name)}/axes`),

        analysisDefaults: (name) =>
            request(`/api/arpes/analysis/${encodeURIComponent(name)}/defaults`),
        analysisFitCurve: (name, payload) =>
            request(`/api/arpes/analysis/${encodeURIComponent(name)}/curve`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        analysisFitStack: (name, payload) =>
            request(`/api/arpes/analysis/${encodeURIComponent(name)}/stack`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        analysisQpResults: (name, payload) =>
            request(`/api/arpes/analysis/${encodeURIComponent(name)}/qp-results`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        analysisGapCurve: (name, payload) =>
            request(`/api/arpes/analysis/${encodeURIComponent(name)}/gap-curve`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        analysisGapStack: (name, payload) =>
            request(`/api/arpes/analysis/${encodeURIComponent(name)}/gap-stack`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        analysisOverlay: async (name, payload) => {
            const response = await fetch(
                `/api/arpes/analysis/${encodeURIComponent(name)}/overlay`,
                {
                    method: "POST",
                    credentials: "same-origin",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }
            );
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* binary error */
                }
                throw new Error(detail);
            }
            const buffer = await response.arrayBuffer();
            const headerLength = new DataView(buffer).getUint32(0, true);
            const header = JSON.parse(
                new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength))
            );
            const floatOffset = 4 + headerLength;
            const n = header.shape[0] * header.shape[1];
            const values = new Float32Array(buffer, floatOffset, n);
            let simValues = null;
            if (header.has_sim && header.sim_offset_floats != null) {
                simValues = new Float32Array(
                    buffer,
                    floatOffset + header.sim_offset_floats * 4,
                    n
                );
            }
            return { header, values, simValues };
        },
        analysisGetNode: (name, node) =>
            request(
                `/api/arpes/analysis/${encodeURIComponent(name)}/${encodeURIComponent(node)}`
            ),

        tensorVolume: async (name, payload) => {
            const response = await fetch(`/api/arpes/${encodeURIComponent(name)}/volume`, {
                method: "POST",
                credentials: "same-origin",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* binary error */
                }
                throw new Error(detail);
            }
            const buffer = await response.arrayBuffer();
            const headerLength = new DataView(buffer).getUint32(0, true);
            const header = JSON.parse(
                new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength))
            );
            const n = header.shape[0] * header.shape[1] * header.shape[2];
            const values = new Float32Array(buffer, 4 + headerLength, n);
            return { header, values };
        },

        /* Unpacks the framed slice: uint32 header length, header JSON, then
           float32 values. The header is padded so the values start aligned
           and can be wrapped without copying. */
        tensorSlice: async (name, payload) => {
            const response = await fetch(`/api/arpes/${encodeURIComponent(name)}/slice`, {
                method: "POST",
                credentials: "same-origin",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* binary endpoint returned no JSON error body */
                }
                throw new Error(detail);
            }

            const buffer = await response.arrayBuffer();
            const headerLength = new DataView(buffer).getUint32(0, true);
            const header = JSON.parse(
                new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength))
            );
            const values = new Float32Array(
                buffer, 4 + headerLength, header.shape[0] * header.shape[1]
            );
            return { header, values };
        },

        dftStructures: () => request("/api/dft/structures"),
        dftSlabPresets: () => request("/api/dft/slab-presets"),
        dftPrepareSlab: (name, payload) =>
            request(`/api/dft/${encodeURIComponent(name)}/prepare-slab`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        dftBands: (name, payload) =>
            request(`/api/dft/${encodeURIComponent(name)}/bands`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        dftIsoenergy: (name, payload) =>
            request(`/api/dft/${encodeURIComponent(name)}/isoenergy`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        dftFatBands: (name, fatTarget = "none") =>
            request(`/api/dft/${encodeURIComponent(name)}/bands/fat`, {
                method: "POST",
                body: JSON.stringify({ fat_target: fatTarget }),
            }),
        dftBzContext: (name) =>
            request(`/api/dft/${encodeURIComponent(name)}/bz-context`),
        dftGapPredict: (name, fidelity = "PBE") =>
            request(`/api/dft/${encodeURIComponent(name)}/gap-predict`, {
                method: "POST",
                body: JSON.stringify({ fidelity }),
            }),
        dftUploadWannier: (name, hrFile, scfFile = null) => {
            const form = new FormData();
            form.append("file", hrFile);
            if (scfFile) form.append("scf_out", scfFile);
            return upload(`/api/dft/${encodeURIComponent(name)}/wannier`, form);
        },
        dftSolvers: () => request("/api/dft/solvers"),
        qeGenerate: (name, payload) =>
            request(`/api/dft/${encodeURIComponent(name)}/qe/generate`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        qeQueue: (name, payload) =>
            request(`/api/dft/${encodeURIComponent(name)}/qe/queue`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        qeJob: (jobId) => request(`/api/dft/jobs/${encodeURIComponent(jobId)}`),
        qeCancel: (jobId) =>
            request(`/api/dft/jobs/${encodeURIComponent(jobId)}/cancel`, {
                method: "POST",
            }),
        /* Binary zip download — returns a Blob the browser can save. */
        qeBundle: async (name, payload) => {
            const response = await fetch(`/api/dft/${encodeURIComponent(name)}/qe/bundle`, {
                method: "POST",
                credentials: "same-origin",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* no JSON body */
                }
                throw new Error(detail);
            }
            return response.blob();
        },

        tensorProfiles: (name, payload) =>
            request(`/api/arpes/${encodeURIComponent(name)}/profiles`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),

        arpesExportFigure: async (name, payload) => {
            const response = await fetch(
                `/api/arpes/${encodeURIComponent(name)}/export/figure`,
                {
                    method: "POST",
                    credentials: "same-origin",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }
            );
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* ignore */
                }
                throw new Error(detail);
            }
            return response.blob();
        },

        arpesSimulate: (payload) =>
            request("/api/arpes/simulate", {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        arpesJob: (jobId) =>
            request(`/api/arpes/jobs/${encodeURIComponent(jobId)}`),
        arpesCancelJob: (jobId) =>
            request(`/api/arpes/jobs/${encodeURIComponent(jobId)}/cancel`, {
                method: "POST",
            }),
        arpesPushJob: (jobId, payload = {}) =>
            request(`/api/arpes/jobs/${encodeURIComponent(jobId)}/push`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
        arpesPreview: async (jobId, eIndex = 0) => {
            const response = await fetch(
                `/api/arpes/jobs/${encodeURIComponent(jobId)}/preview?e_index=${eIndex}`,
                { credentials: "same-origin" }
            );
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    if (body.detail) detail = body.detail;
                } catch (err) {
                    /* ignore */
                }
                throw new Error(detail);
            }
            const buffer = await response.arrayBuffer();
            const headerLength = new DataView(buffer).getUint32(0, true);
            const header = JSON.parse(
                new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength))
            );
            const values = new Float32Array(
                buffer, 4 + headerLength, header.shape[0] * header.shape[1]
            );
            return { header, values };
        },
    };
})();
