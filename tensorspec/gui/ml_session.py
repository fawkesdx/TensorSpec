"""Shared state for the ML suite.

The ML tabs used to reach directly into each other's widgets: activate_data
repopulated combo boxes belonging to the clustering, active-learning and
simulate-AL tabs, and on_cluster_finish did the same. MLSession replaces those
writes with signals so each panel only ever touches its own widgets.
"""
from PySide6.QtCore import QObject, Signal

from tensorspec.core.data_models import TensorData


class MLSession(QObject):
    """Workspace and active-dataset state shared by every ML panel."""

    # A dataset was added to or removed from the in-memory workspace.
    workspace_changed = Signal()
    # The user selected a dataset to work on; payload is that dataset.
    data_activated = Signal(object)
    # The "embeddings_*" keys available on the active dataset changed.
    embeddings_changed = Signal(list)
    # The "domains_*" keys available on the active dataset changed.
    domains_changed = Signal(list)
    # Progress value (0-100) and message for the suite status bar.
    status_changed = Signal(int, str)

    EMBEDDING_PREFIX = "embeddings_"
    DOMAIN_PREFIX = "domains_"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.workspace = {}
        self.current_folder = ""
        self.current_view_data = None
        # Assigned by MLSuite once the shared DataViewerPanel exists.
        self.viewer = None

    def set_status(self, value, message):
        self.status_changed.emit(value, message)

    def add_dataset(self, name, data):
        self.workspace[name] = data
        self.workspace_changed.emit()

    def remove_dataset(self, name):
        if name in self.workspace:
            del self.workspace[name]
            self.workspace_changed.emit()

    def activate(self, data):
        self.current_view_data = data
        self.data_activated.emit(data)

    def _keys_with_prefix(self, prefix):
        data = self.current_view_data
        if data is None:
            return []
        if isinstance(data, TensorData):
            keys = (data.metadata.get("layers") or {}).keys()
        elif isinstance(data, dict):
            keys = data.keys()
        else:
            return []
        return [k for k in keys if k.startswith(prefix)]

    def embedding_keys(self):
        return self._keys_with_prefix(self.EMBEDDING_PREFIX)

    def domain_keys(self):
        return self._keys_with_prefix(self.DOMAIN_PREFIX)

    def notify_embeddings(self):
        self.embeddings_changed.emit(self.embedding_keys())

    def notify_domains(self):
        self.domains_changed.emit(self.domain_keys())
