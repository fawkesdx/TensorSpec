from __future__ import annotations

from tensorspec.core.io.loaders.maestro.types import ScanLoop, ScanMotor, ScanPlan


def parse_low_level_scan(header_table) -> ScanPlan:
    fields = _index_header_table(header_table)

    mode_name = _decode_string(fields.get("lwlvnm", ""))
    parallel = _decode_parallel(fields.get("scanpar"))
    num_loops = int(_decode_string(fields.get("lwlvlpn", "0")) or "0")

    loops: list[ScanLoop] = []
    for loop_idx in range(num_loops):
        motor_count = int(
            _decode_string(fields.get(f"nmsbdv{loop_idx}", "0")) or "0"
        )
        motors: list[ScanMotor] = []
        for motor_idx in range(motor_count):
            prefix = f"{loop_idx}_{motor_idx}"
            motors.append(
                ScanMotor(
                    name=_decode_string(fields.get(f"nm_{prefix}", "")),
                    units=_decode_string(fields.get(f"un_{prefix}", "")),
                    start=float(_decode_string(fields.get(f"st_{prefix}", "0")) or "0"),
                    end=float(_decode_string(fields.get(f"en_{prefix}", "0")) or "0"),
                    n=int(_decode_string(fields.get(f"n_{prefix}", "0")) or "0"),
                )
            )
        loops.append(ScanLoop(motors=motors))

    return ScanPlan(mode_name=mode_name, loops=loops, parallel=parallel)


def _index_header_table(header_table) -> dict[str, str]:
    fields: dict[str, str] = {}
    for row in header_table:
        longname, name, value, _comment = row
        tag = _decode_text(longname or name).casefold().strip()
        fields[tag] = value
    return fields


def _decode_parallel(value) -> bool:
    if value is None:
        return False
    text = _decode_string(value).casefold()
    return text in {"t", "true", "p", "parallel", "y", "yes"}


def _decode_string(value) -> str:
    text = _decode_text(value).strip()
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1]
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1]
    return text


def _decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
