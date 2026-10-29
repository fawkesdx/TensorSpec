"""xarray Datasets for the ARPES peak-fit outputs under ``/analysis/<kind>`` (no Qt).

As in core/raman/products.py every Dataset has a ``data`` variable so ``pull_tensor_data`` and
the generic DataViewer can open it: here ``data`` is the stack of fitted lines on
``(line axis, fit axis)`` -- ``(Energy, kx)`` for an MDC series, ``(kx, Energy)`` for EDCs --
next to ``model`` and ``background``. The per-band parameters sit on ``(line axis, band)``.
Components are not stored; :func:`line_components` recomputes them from the parameters.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import xarray as xr

from tensorspec.core.arpes.peakfit import (
    BandModel,
    BandSeries,
    FitSettings,
    fermi_function,
    peak_profile,
)

#: /analysis/<kind> products of the fit explorer.
PEAKFIT_ANALYSIS_KINDS = ("mdc_peakfit", "edc_peakfit")
#: Per-band variables on (line, band).
BAND_VARIABLES = ("centre", "centre_err", "fwhm", "fwhm_err", "height", "area")


def _attr(v: Any):
    if isinstance(v, np.generic):
        return v.item()
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    return json.dumps(v, default=str)


def peakfit_kind(direction: str) -> str:
    return "mdc_peakfit" if direction == "mdc" else "edc_peakfit"


def peakfit_dataset(series: BandSeries, band_model: BandModel, settings: FitSettings, *,
                    resolution_source: str = "", extra_attrs: dict | None = None) -> xr.Dataset:
    """One Dataset from a fitted series.

    Dims: ``(<line axis>, band)`` for ``centre`` / ``centre_err`` / ``fwhm`` / ``fwhm_err`` /
    ``height`` / ``area``; ``(<line axis>, <fit axis>)`` for ``data`` / ``model`` /
    ``background`` (NaN where a line was not fitted); ``chi2``, ``r_squared`` and ``fitted`` per
    line. Attrs: ``band_model`` and ``settings`` as JSON, ``resolution_fwhm``,
    ``resolution_source``, ``direction``, ``kind``, ``line_axis``, ``fit_axis``.
    """
    line_axis = series.across_label or ("Energy" if series.direction == "mdc" else "kx")
    fit_axis = series.along_label or ("kx" if series.direction == "mdc" else "Energy")
    if line_axis == fit_axis:
        fit_axis = fit_axis + "_fit"
    along = np.asarray(series.along, dtype=float)
    n_lines, n_bands = series.centres.shape

    stack = {k: np.full((n_lines, along.size), np.nan) for k in ("data", "model", "background")}
    fitted = np.zeros(n_lines, dtype=bool)
    r_squared = np.full(n_lines, np.nan)
    for row, fit in enumerate(series.fits or []):
        if fit is None:
            continue
        # fit.x is the finite, sorted subset of the along axis: put each point back by value
        idx = np.searchsorted(along, fit.x) if np.all(np.diff(along) > 0) else \
            np.array([int(np.argmin(np.abs(along - xv))) for xv in fit.x])
        idx = np.clip(idx, 0, along.size - 1)
        stack["data"][row, idx] = fit.y
        stack["model"][row, idx] = fit.model
        stack["background"][row, idx] = fit.background
        fitted[row] = True
        r_squared[row] = fit.r_squared

    coords = {
        line_axis: (line_axis, np.asarray(series.positions, float), {"units": series.across_unit}),
        fit_axis: (fit_axis, along, {"units": series.along_unit}),
        "band": ("band", np.array(list(series.names), dtype=object)),
    }
    unit = series.along_unit
    vars_ = {
        "data": ((line_axis, fit_axis), stack["data"],
                 {"long_name": f"{series.direction.upper()} lines as fitted"}),
        "model": ((line_axis, fit_axis), stack["model"], {"long_name": "total fit model"}),
        "background": ((line_axis, fit_axis), stack["background"], {"long_name": "fit background"}),
        "centre": ((line_axis, "band"), np.asarray(series.centres, float), {"units": unit}),
        "centre_err": ((line_axis, "band"), np.asarray(series.centre_errors, float), {"units": unit}),
        "fwhm": ((line_axis, "band"), np.asarray(series.widths, float), {"units": unit}),
        "fwhm_err": ((line_axis, "band"), np.asarray(series.width_errors, float), {"units": unit}),
        "height": ((line_axis, "band"), np.asarray(series.heights, float)),
        "area": ((line_axis, "band"), np.asarray(series.areas, float)),
        "chi2": ((line_axis,), np.asarray(series.chi2, float), {"long_name": "reduced chi2"}),
        "r_squared": ((line_axis,), r_squared),
        "fitted": ((line_axis,), fitted),
    }
    attrs = {
        "kind": peakfit_kind(series.direction),
        "direction": series.direction,
        "line_axis": line_axis,
        "fit_axis": fit_axis,
        "band_model": band_model.to_json(),
        "settings": settings.to_json(),
        "resolution_fwhm": float(settings.resolution),
        "resolution_source": str(resolution_source or ""),
        "n_lines": int(n_lines),
        "n_fitted": int(fitted.sum()),
        "band_names": json.dumps(list(series.names)),
    }
    attrs.update({k: _attr(v) for k, v in (extra_attrs or {}).items()})
    return xr.Dataset(vars_, coords=coords, attrs=attrs)


def is_peakfit_dataset(ds) -> bool:
    try:
        return str(ds.attrs.get("kind", "")) in PEAKFIT_ANALYSIS_KINDS and "centre" in ds
    except Exception:
        return False


def band_model_of(ds: xr.Dataset) -> BandModel:
    return BandModel.from_json(str(ds.attrs["band_model"]))


def settings_of(ds: xr.Dataset) -> FitSettings:
    return FitSettings.from_json(str(ds.attrs["settings"]))


def nearest_line(ds: xr.Dataset, position: float) -> int:
    """Index of the fitted line nearest ``position`` on the line axis."""
    axis = np.asarray(ds.coords[str(ds.attrs["line_axis"])].values, float)
    return int(np.argmin(np.abs(axis - float(position))))


def line_components(ds: xr.Dataset, index: int) -> list:
    """The components of line ``index``, recomputed from the stored parameters:
    ``[(band name, band index, y on the fit axis), ...]`` on top of the background, each the
    height times its unit profile (times the Fermi-Dirac occupation for an EDC fit with it
    on), for every band that was fitted on that line."""
    model = band_model_of(ds)
    settings = settings_of(ds)
    x = np.asarray(ds.coords[str(ds.attrs["fit_axis"])].values, float)
    background = np.nan_to_num(np.asarray(ds["background"].values[index], float))
    occupation = (fermi_function(x, settings.ef, settings.temperature)
                  if ds.attrs.get("direction") == "edc" and settings.fermi else 1.0)
    out = []
    for j, band in enumerate(model.bands):
        if j >= ds.sizes["band"]:
            break
        centre = float(ds["centre"].values[index, j])
        if not np.isfinite(centre):
            continue
        fwhm = float(ds["fwhm"].values[index, j])
        height = float(ds["height"].values[index, j])
        profile = height * peak_profile(band.shape, x, centre, fwhm, settings.resolution)
        out.append((band.name, j, background + profile * occupation))
    return out


def line_table(ds: xr.Dataset, index: int) -> list:
    """Rows ``(band, centre, centre_err, fwhm, height)`` of line ``index`` for a table."""
    rows = []
    for j, name in enumerate(ds.coords["band"].values):
        centre = float(ds["centre"].values[index, j])
        if not np.isfinite(centre):
            continue
        rows.append((str(name), centre, float(ds["centre_err"].values[index, j]),
                     float(ds["fwhm"].values[index, j]), float(ds["height"].values[index, j])))
    return rows
