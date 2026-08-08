import os

from .loaders.mock_data_loader import MockDataLoader
from .loaders.maestro_loader import MaestroLoader, lookup_measurement_log
from tensorspec.core.data_models import TensorData


class ARPESLoader:
    """
    Main agnostic IO Manager.
    Passes the file through loaders, then packages it into the universal TensorData format.
    """

    @classmethod
    def load(cls, filepath, measurement_log_path=None, measurement_log_row=None):
        filepath_str = str(filepath)
        filename = os.path.basename(filepath_str)

        log_row = measurement_log_row
        if log_row is None:
            log_row = lookup_measurement_log(filepath_str, measurement_log_path)

        attempts = [
            ("MockDataLoader", lambda: MockDataLoader(filepath_str).load()),
            (
                "MaestroLoader",
                lambda: MaestroLoader(
                    filepath_str,
                    measurement_log_row=log_row,
                    measurement_log_path=measurement_log_path,
                ).load(),
            ),
        ]

        last_error = None
        for name, attempt in attempts:
            try:
                raw_dict = attempt()
            except ValueError as exc:
                last_error = exc
                continue
            except Exception as exc:
                print(f"Warning: {name} failed on '{filename}': {exc}")
                last_error = exc
                continue

            axes_dict = raw_dict.get("axes", {})
            labels = list(axes_dict.keys())
            axes = list(axes_dict.values())
            axis_units = raw_dict.get("axis_units")
            if axis_units is None or len(axis_units) != len(labels):
                axis_units = [
                    "eV"
                    if "eV" in label or "Energy" in label
                    else "deg"
                    if "deg" in label or "Angle" in label or "Defl" in label
                    else "um"
                    if any(tok in label for tok in ("Sample", "Scan", "Stage"))
                    else "px"
                    if "Pixel" in label
                    else "a.u."
                    for label in labels
                ]

            metadata = dict(raw_dict.get("metadata") or {})
            metadata.setdefault("Facility", raw_dict.get("facility", "Unknown"))
            if raw_dict.get("measurement_kind"):
                metadata.setdefault("Measurement_Type", raw_dict["measurement_kind"])

            return TensorData(
                value=raw_dict["data"],
                axes=axes,
                labels=labels,
                units=list(axis_units),
                data_type=raw_dict.get("mode", "ARPES"),
                metadata=metadata,
            )

        detail = f" ({last_error})" if last_error else ""
        raise ValueError(f"Could not load '{filename}': No matching facility format found.{detail}")
