/* Renders the workspace variable tree and drives the metadata inspector.
   Mirrors refresh_workspace_tree() and on_item_selected() from the Qt browser. */

const WorkspaceTree = (() => {
    let selected = null;

    const el = {};

    function cacheElements() {
        el.tbody = document.querySelector("[data-tree-body]");
        el.empty = document.querySelector("[data-tree-empty]");
        el.refresh = document.querySelector("[data-action='refresh']");
        el.seed = document.querySelector("[data-action='seed-demo']");
        el.inspector = document.querySelector("[data-inspector]");
        el.inspectorBadge = document.querySelector("[data-inspector-badge]");
        el.launch = document.querySelector("[data-action='launch-viewer']");
        el.count = document.querySelector("[data-variable-count]");
        el.status = document.querySelector("[data-connection-status]");
    }

    function setStatus(text) {
        if (el.status) el.status.textContent = text;
    }

    function renderRows(items) {
        el.tbody.textContent = "";

        items.forEach((item) => {
            const row = document.createElement("tr");
            row.setAttribute("aria-selected", String(item.name === selected));
            row.dataset.name = item.name;
            row.tabIndex = 0;

            [
                ["tree__name", item.name],
                ["tree__type", item.type],
                ["tree__dims", item.dims],
            ].forEach(([className, text]) => {
                const cell = document.createElement("td");
                cell.className = className;
                cell.textContent = text;
                row.appendChild(cell);
            });

            row.addEventListener("click", () => select(item.name));
            row.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    select(item.name);
                }
            });

            el.tbody.appendChild(row);
        });

        if (el.empty) el.empty.hidden = items.length > 0;
        if (el.count) {
            el.count.textContent =
                items.length === 1 ? "1 variable loaded" : `${items.length} variables loaded`;
        }
    }

    function renderInspector(detail) {
        const lines = [
            `Variable: ${detail.name}`,
            `Type: ${detail.type}`,
            `Shape: ${detail.dims}`,
            "-".repeat(40),
        ];

        const entries = Object.entries(detail.metadata || {});
        if (entries.length === 0) {
            lines.push("No metadata recorded for this object.");
        } else {
            entries.forEach(([key, value]) => lines.push(`${key}: ${value}`));
        }

        el.inspector.textContent = lines.join("\n");
        el.inspector.classList.remove("metadata__placeholder");
        if (el.inspectorBadge) el.inspectorBadge.textContent = detail.name;
        if (el.launch) el.launch.disabled = !detail.viewable;
    }

    function showPlaceholder(message) {
        el.inspector.textContent = message;
        el.inspector.classList.add("metadata__placeholder");
        if (el.inspectorBadge) el.inspectorBadge.textContent = "nothing selected";
        if (el.launch) el.launch.disabled = true;
    }

    async function select(name) {
        selected = name;
        el.tbody.querySelectorAll("tr").forEach((row) => {
            row.setAttribute("aria-selected", String(row.dataset.name === name));
        });

        try {
            renderInspector(await TensorSpecAPI.getItem(name));
        } catch (error) {
            showPlaceholder(`Could not read '${name}': ${error.message}`);
        }
    }

    async function refresh() {
        try {
            const listing = await TensorSpecAPI.listItems();
            renderRows(listing.items);
            setStatus("Connected to core");

            if (listing.items.length === 0) {
                selected = null;
                showPlaceholder("Workspace is empty. Load data or seed the demo fixture.");
            } else if (selected && listing.items.some((item) => item.name === selected)) {
                await select(selected);
            } else {
                await select(listing.items[0].name);
            }
        } catch (error) {
            setStatus("Server unreachable");
            showPlaceholder(
                `Cannot reach the TensorSpec service: ${error.message}\n\n` +
                    "Start it with:\n  uvicorn tensorspec.web.server.app:app --reload"
            );
        }
    }

    async function seedDemo() {
        el.seed.disabled = true;
        try {
            await TensorSpecAPI.seedDemo({});
            await refresh();
        } catch (error) {
            showPlaceholder(`Seeding failed: ${error.message}`);
        } finally {
            el.seed.disabled = false;
        }
    }

    function init() {
        cacheElements();
        if (!el.tbody) return;

        if (el.refresh) el.refresh.addEventListener("click", refresh);
        if (el.seed) el.seed.addEventListener("click", seedDemo);
        refresh();
    }

    return { init, refresh };
})();

document.addEventListener("DOMContentLoaded", WorkspaceTree.init);
