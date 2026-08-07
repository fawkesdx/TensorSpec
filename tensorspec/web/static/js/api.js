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

        tensorAxes: (name) =>
            request(`/api/arpes/${encodeURIComponent(name)}/axes`),

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

        tensorProfiles: (name, payload) =>
            request(`/api/arpes/${encodeURIComponent(name)}/profiles`, {
                method: "POST",
                body: JSON.stringify(payload),
            }),
    };
})();
