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

    return {
        health: () => request("/api/health"),
        listItems: () => request("/api/workspace/items"),
        getItem: (name) => request(`/api/workspace/items/${encodeURIComponent(name)}`),
        seedDemo: (payload = {}) =>
            request("/api/workspace/demo", {
                method: "POST",
                body: JSON.stringify(payload),
            }),
    };
})();
