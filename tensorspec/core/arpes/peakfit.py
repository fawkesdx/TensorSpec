"""MDC / EDC peak fitting for the ARPES suite (no Qt, no plotting).

A port of ``tools/peaks.py`` (and the few helpers it uses from ``tools/fermi.py`` and
``tools/process.py``) from Ding Pei's ARPES-data-browser
(https://github.com/DingPei1995/ARPES-data-browser, vendored commit f7dccbf), brought into
``tensorspec.core`` so the fit explorer does not depend on a checkout outside the repository.
The algorithm is his; the changes here are the serialisable ``BandModel`` / ``FitSettings``,
axis labels carried through the series, and ``fwhm`` as the seed width name.

What the fitter does, in his words:

* **Seeding.** A band is marked at a few line positions and the rest is interpolated with
  PCHIP (monotone, cannot overshoot between sparse seeds); outside the seeded range the
  nearest seed is held. A band exists only over the range its seeds span.
* **Variable projection.** Only centres and widths go to the bounded ``least_squares``; the
  heights (held >= 0) and the background are solved exactly by linear least squares at every
  step.
* **Bounds.** Each centre may move at most 1.5 widths from its seed, which stops two bands
  swapping identity at a crossing.
* **Errors** from the full Jacobian at the solution with Poisson weights; reduced chi2 per line.
* **EDCs:** the Fermi-Dirac occupation multiplies each peak *before* the Gaussian resolution
  convolution; the background is not occupation-modulated.
* ``fit_series`` runs two passes: the first seeded from the band model, the second from the
  first pass's own answer, which follows the fitted dispersion densely.

Widths are FWHM throughout. For Voigt peaks the Gaussian FWHM is the resolution and the
Lorentzian FWHM is the fitted width.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import optimize
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, wofz

#: Line shapes a band can take.
PEAK_SHAPES = ("lorentzian", "gaussian", "voigt")
#: Backgrounds that can sit under the peaks of one line.
BACKGROUNDS = ("none", "constant", "linear", "shirley")
#: Series directions: lines along momentum at each energy, or along energy at each momentum.
DIRECTIONS = ("mdc", "edc")
#: Boltzmann's constant in eV/K.
K_B = 8.617333262e-5
#: FWHM = SIGMA_TO_FWHM * sigma for a Gaussian.
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ==========================================================================
# Helpers ported from tools/fermi.py and tools/process.py
# ==========================================================================
def fermi_function(energy, ef: float, temperature: float) -> np.ndarray:
    """Fermi-Dirac occupation; ``expit`` so a large exponent does not overflow."""
    energy = np.asarray(energy, dtype=float)
    kt = K_B * max(float(temperature), 1e-6)
    return expit(-(energy - float(ef)) / kt)


def _convolution_offsets(sigma: float, kt: float, n_sigma: float = 6.0,
                         max_points: int = 1201):
    """Offsets and weights for integrating against a Gaussian of width ``sigma``; the step
    follows the narrower of the resolution and the thermal width."""
    scale = min(sigma, kt) if kt > 0 else sigma
    step = max(scale / 6.0, 2.0 * n_sigma * sigma / (max_points - 1))
    half = int(np.ceil(n_sigma * sigma / step))
    offsets = np.arange(-half, half + 1) * step
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    return offsets, weights / weights.sum()


def shirley_background(energy, spectrum, *, iterations: int = 30, tol: float = 1e-6):
    """The Shirley inelastic background of one spectrum: at each point proportional to the
    spectral weight above it, iterated to convergence, ending at each endpoint's own level."""
    energy = np.asarray(energy, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    order = np.argsort(energy)
    e = energy[order]
    y = np.nan_to_num(spectrum[order], nan=0.0)
    lo, hi = float(y[0]), float(y[-1])
    background = np.full(y.shape, hi)
    for _ in range(max(1, int(iterations))):
        above = y - background
        integral = np.concatenate(([0.0], np.cumsum((above[1:] + above[:-1]) / 2.0 * np.diff(e))))
        total = integral[-1]
        if abs(total) < 1e-30:
            break
        new = hi + (lo - hi) * (1.0 - integral / total)
        if np.max(np.abs(new - background)) < tol * max(abs(hi - lo), 1e-12):
            background = new
            break
        background = new
    out = np.empty_like(background)
    out[order] = background
    return out


# ==========================================================================
# Line shapes
# ==========================================================================
def peak_profile(shape: str, x, centre: float, width: float, resolution: float = 0.0):
    """One unit-height peak. ``width`` is the FWHM; ``resolution`` the Gaussian FWHM it is
    convolved with (a Voigt uses the Faddeeva function, normalised to unit height)."""
    x = np.asarray(x, dtype=float)
    width = max(abs(float(width)), 1e-12)
    resolution = max(float(resolution), 0.0)

    if shape == "gaussian":
        full = float(np.hypot(width, resolution))
        sigma = full / SIGMA_TO_FWHM
        return np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    if shape == "lorentzian" and resolution <= 0:
        half = width / 2.0
        return half ** 2 / ((x - centre) ** 2 + half ** 2)
    if shape not in PEAK_SHAPES:
        raise ValueError(f"shape {shape!r} is not one of {PEAK_SHAPES}")

    sigma = max(resolution, 1e-12) / SIGMA_TO_FWHM
    gamma = width / 2.0
    scale = sigma * np.sqrt(2.0)
    profile = np.real(wofz(((x - centre) + 1j * gamma) / scale))
    peak = float(np.real(wofz(1j * gamma / scale)))
    return profile / max(peak, 1e-30)


def _area_factor(shape: str, width: float, resolution: float) -> float:
    """Integral of a unit-height peak of this shape and width."""
    if shape == "gaussian":
        full = float(np.hypot(width, resolution))
        return full * np.sqrt(np.pi / (4.0 * np.log(2.0)))
    if shape == "lorentzian" and resolution <= 0:
        return float(np.pi * width / 2.0)
    grid = np.linspace(-20.0 * (width + resolution), 20.0 * (width + resolution), 4001)
    return float(np.trapezoid(peak_profile(shape, grid, 0.0, width, resolution), grid))


# ==========================================================================
# Bands, seeds, settings (all JSON-serialisable)
# ==========================================================================
@dataclass
class Seed:
    """One hand-placed point: this band is at ``centre`` along the line at ``position``.

    ``position`` is on the axis the lines are spaced along (energy for MDCs, momentum for
    EDCs); ``centre`` and ``fwhm`` are along the line."""
    position: float
    centre: float
    fwhm: float
    height: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "Seed":
        return cls(float(d["position"]), float(d["centre"]),
                   float(d.get("fwhm", d.get("width", 0.0))), float(d.get("height", 0.0)))


@dataclass
class Band:
    """One band across the whole series. ``share_width`` names another band whose width this
    one copies (the two branches of a symmetric dispersion); ``fix_width`` holds the seed
    width. ``colour`` is an index into the explorer's palette (-1 = the band's own index)."""
    name: str
    seeds: list = field(default_factory=list)
    shape: str = "lorentzian"
    share_width: str = ""
    fix_width: bool = False
    colour: int = -1

    def __post_init__(self):
        if self.shape not in PEAK_SHAPES:
            raise ValueError(f"shape {self.shape!r} is not one of {PEAK_SHAPES}")
        self.seeds = [s if isinstance(s, Seed) else Seed.from_dict(s) if isinstance(s, dict)
                      else Seed(*s) for s in self.seeds]

    def to_dict(self) -> dict:
        return {"name": self.name, "shape": self.shape, "share_width": self.share_width,
                "fix_width": bool(self.fix_width), "colour": int(self.colour),
                "seeds": [asdict(s) for s in self.sorted_seeds()]}

    @classmethod
    def from_dict(cls, d: dict) -> "Band":
        return cls(str(d["name"]), [Seed.from_dict(s) for s in d.get("seeds", [])],
                   str(d.get("shape", "lorentzian")), str(d.get("share_width", "") or ""),
                   bool(d.get("fix_width", False)), int(d.get("colour", -1)))

    def sorted_seeds(self):
        return sorted(self.seeds, key=lambda s: s.position)

    def span(self):
        """The range of lines this band exists over, from its own seeds."""
        if not self.seeds:
            return None
        positions = [s.position for s in self.seeds]
        return (min(positions), max(positions))

    def covers(self, position: float, tol: float = 0.0) -> bool:
        span = self.span()
        if span is None:
            return False
        return (span[0] - tol) <= float(position) <= (span[1] + tol)

    def guess(self, positions):
        """Starting centre, width and height at each of ``positions`` (PCHIP between seeds,
        held at the nearest seed outside them)."""
        positions = np.atleast_1d(np.asarray(positions, dtype=float))
        seeds = self.sorted_seeds()
        if not seeds:
            raise ValueError(f"band {self.name!r} has no seeds")
        known = np.array([s.position for s in seeds], dtype=float)
        out = []
        for values in (np.array([s.centre for s in seeds], dtype=float),
                       np.array([s.fwhm for s in seeds], dtype=float),
                       np.array([s.height for s in seeds], dtype=float)):
            if known.size == 1:
                out.append(np.full(positions.shape, float(values[0])))
                continue
            curve = PchipInterpolator(known, values, extrapolate=False)
            out.append(np.asarray(curve(np.clip(positions, known[0], known[-1])), dtype=float))
        return out[0], out[1], out[2]


@dataclass
class BandModel:
    """The bands of one fit, as stored in ``attrs['band_model']`` of a peak-fit product."""
    bands: list = field(default_factory=list)

    def __post_init__(self):
        self.bands = [b if isinstance(b, Band) else Band.from_dict(b) for b in self.bands]
        names = [b.name for b in self.bands]
        if len(set(names)) != len(names):
            raise ValueError(f"band names must be unique: {names}")

    @property
    def names(self) -> list:
        return [b.name for b in self.bands]

    def to_dict(self) -> dict:
        return {"bands": [b.to_dict() for b in self.bands]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d) -> "BandModel":
        if isinstance(d, (list, tuple)):
            return cls([Band.from_dict(b) if isinstance(b, dict) else b for b in d])
        return cls([Band.from_dict(b) for b in d.get("bands", [])])

    @classmethod
    def from_json(cls, text: str) -> "BandModel":
        return cls.from_dict(json.loads(text))


@dataclass
class FitSettings:
    """Everything about the fit that is not a band. ``resolution`` is the Gaussian FWHM in the
    units of the fitted axis (eV for EDCs; for MDCs the energy resolution divided by v_F)."""
    background: str = "linear"
    resolution: float = 0.0
    weighting: str = "poisson"
    #: EDC only: multiply the model by the Fermi-Dirac occupation.
    fermi: bool = False
    ef: float = 0.0
    temperature: float = 30.0
    max_iterations: int = 400

    def __post_init__(self):
        if self.background not in BACKGROUNDS:
            raise ValueError(f"background {self.background!r} is not one of {BACKGROUNDS}")
        if self.weighting not in ("poisson", "none"):
            raise ValueError(f"weighting {self.weighting!r} is not poisson or none")

    def background_columns(self) -> int:
        return {"none": 0, "shirley": 0, "constant": 1, "linear": 2}[self.background]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "FitSettings":
        return cls(str(d.get("background", "linear")), float(d.get("resolution", 0.0)),
                   str(d.get("weighting", "poisson")), bool(d.get("fermi", False)),
                   float(d.get("ef", 0.0)), float(d.get("temperature", 30.0)),
                   int(d.get("max_iterations", 400)))

    @classmethod
    def from_json(cls, text: str) -> "FitSettings":
        return cls.from_dict(json.loads(text))


# ==========================================================================
# One line
# ==========================================================================
@dataclass
class LineFit:
    """One fitted MDC or EDC."""
    x: np.ndarray
    y: np.ndarray
    model: np.ndarray
    background: np.ndarray
    names: list
    centres: np.ndarray
    centre_errors: np.ndarray
    widths: np.ndarray
    width_errors: np.ndarray
    heights: np.ndarray
    height_errors: np.ndarray
    areas: np.ndarray
    position: float = float("nan")
    chi2: float = float("nan")
    r_squared: float = float("nan")
    shapes: list = field(default_factory=list)
    resolution: float = 0.0
    success: bool = True
    message: str = ""

    @property
    def residual(self) -> np.ndarray:
        return self.y - self.model

    def component(self, index: int) -> np.ndarray:
        """One band's own peak, on top of the background (not occupation-modulated)."""
        return self.background + self.heights[index] * peak_profile(
            self.shapes[index], self.x, self.centres[index], self.widths[index], self.resolution)


def _weights(y, weighting: str):
    if weighting == "poisson":
        return np.sqrt(np.maximum(np.abs(y), 1.0))
    if weighting == "none":
        return np.ones_like(y)
    raise ValueError(f"weighting {weighting!r} is not poisson or none")


def _occupation(x, settings: FitSettings):
    """``(offsets, weights, f)`` for the resolution/Fermi convolution, or Nones without it."""
    x = np.asarray(x, dtype=float)
    sigma = max(float(settings.resolution), 0.0) / SIGMA_TO_FWHM
    if not settings.fermi:
        return None, None, None
    if sigma <= 0:
        return (np.zeros(1), np.ones(1),
                fermi_function(x, settings.ef, settings.temperature)[:, None])
    kt = K_B * max(float(settings.temperature), 1e-6)
    offsets, weights = _convolution_offsets(sigma, kt)
    grid = x[:, None] - offsets[None, :]
    return offsets, weights, fermi_function(grid, settings.ef, settings.temperature)


def _design(x, centres, widths, shapes, settings: FitSettings, fermi_cache):
    """The linear model's columns: one per band, then the background."""
    x = np.asarray(x, dtype=float)
    columns = []
    offsets, weights, occupation = fermi_cache
    for centre, width, shape in zip(centres, widths, shapes):
        if occupation is None:
            columns.append(peak_profile(shape, x, centre, width, settings.resolution))
        else:
            grid = x[:, None] - offsets[None, :]
            profile = peak_profile(shape, grid, centre, width, 0.0)
            columns.append((profile * occupation) @ weights)
    n_background = settings.background_columns()
    if n_background >= 1:
        columns.append(np.ones_like(x))
    if n_background >= 2:
        columns.append(x - float(np.mean(x)))
    return np.column_stack(columns) if columns else np.zeros((x.size, 0))


def _solve_linear(design, y, sigma, n_peaks):
    """Heights (>= 0) and background (free) by weighted linear least squares."""
    if design.shape[1] == 0:
        return np.zeros(0)
    scaled = design / sigma[:, None]
    target = y / sigma
    lower = np.concatenate([np.zeros(n_peaks), np.full(design.shape[1] - n_peaks, -np.inf)])
    upper = np.full(design.shape[1], np.inf)
    try:
        return optimize.lsq_linear(scaled, target, bounds=(lower, upper), method="bvls").x
    except Exception:
        return np.linalg.lstsq(scaled, target, rcond=None)[0]


def _numeric_jacobian(function, params, relative: float = 1e-6):
    params = np.asarray(params, dtype=float)
    base = function(params)
    jacobian = np.empty((base.size, params.size))
    for i in range(params.size):
        step = relative * max(abs(params[i]), 1.0)
        forward, backward = params.copy(), params.copy()
        forward[i] += step
        backward[i] -= step
        jacobian[:, i] = (function(forward) - function(backward)) / (2 * step)
    return jacobian


def fit_line(x, y, bands, settings: FitSettings, *, guesses=None,
             position: float = float("nan"), sigma=None) -> LineFit:
    """Fit one MDC or EDC with the given bands.

    ``guesses`` is an optional ``{name: (centre, width, height)}`` from the seeds or the
    previous line; without it each band's own seeds are interpolated to ``position``.
    ``sigma``, if given, is each point's standard deviation and replaces the weighting.
    """
    bands = list(bands.bands) if isinstance(bands, BandModel) else list(bands)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        good &= np.isfinite(sigma) & (sigma > 0)
    if good.sum() < 3 * len(bands) + 2:
        raise ValueError("not enough valid points for that many bands")
    order = np.argsort(x[good])
    xf, yf = x[good][order], y[good][order]

    shapes = [b.shape for b in bands]
    names = [b.name for b in bands]
    span = float(abs(xf[-1] - xf[0])) or 1.0
    step = span / max(xf.size - 1, 1)

    fixed = np.zeros_like(yf)
    if settings.background == "shirley":
        fixed = shirley_background(xf, yf)

    centres0, widths0 = [], []
    for band in bands:
        if guesses and band.name in guesses:
            centre, width, _ = guesses[band.name]
        else:
            centre, width, _ = (float(v[0]) for v in band.guess([position]))
        centres0.append(float(centre))
        widths0.append(max(float(width), step))

    width_owner = {}
    for i, band in enumerate(bands):
        if band.share_width and band.share_width in names:
            width_owner[i] = names.index(band.share_width)
        else:
            width_owner[i] = i
    free_widths = [i for i, band in enumerate(bands)
                   if width_owner[i] == i and not band.fix_width]

    params0, lower, upper = [], [], []
    for i in range(len(bands)):
        params0.append(centres0[i])
        lower.append(centres0[i] - 1.5 * widths0[i])
        upper.append(centres0[i] + 1.5 * widths0[i])
    for i in free_widths:
        params0.append(widths0[i])
        lower.append(step / 2.0)
        upper.append(5.0 * span)
    params0 = np.asarray(params0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    params0 = np.clip(params0, lower + 1e-12, upper - 1e-12)

    if sigma is None:
        sigma = _weights(yf, settings.weighting)
    else:
        sigma = sigma[good][order]
    fermi_cache = _occupation(xf, settings)
    n_bands = len(bands)

    def unpack(params):
        centres = np.asarray(params[:n_bands], dtype=float)
        widths = np.array(widths0, dtype=float)
        for slot, i in enumerate(free_widths):
            widths[i] = params[n_bands + slot]
        for i in range(n_bands):
            widths[i] = widths[width_owner[i]]
        return centres, widths

    def residual(params):
        centres, widths = unpack(params)
        design = _design(xf, centres, widths, shapes, settings, fermi_cache)
        coeffs = _solve_linear(design, yf - fixed, sigma, n_bands)
        return (design @ coeffs + fixed - yf) / sigma

    result = optimize.least_squares(residual, params0, bounds=(lower, upper),
                                    max_nfev=settings.max_iterations * 10)
    centres, widths = unpack(result.x)
    design = _design(xf, centres, widths, shapes, settings, fermi_cache)
    coeffs = _solve_linear(design, yf - fixed, sigma, n_bands)
    model = design @ coeffs + fixed
    heights = np.asarray(coeffs[:n_bands], dtype=float)
    background = fixed + (design[:, n_bands:] @ coeffs[n_bands:]
                          if design.shape[1] > n_bands else 0.0)

    full = np.concatenate([centres, [widths[i] for i in free_widths], heights, coeffs[n_bands:]])

    def full_model(params):
        c = np.asarray(params[:n_bands], dtype=float)
        w = np.array(widths0, dtype=float)
        for slot, i in enumerate(free_widths):
            w[i] = params[n_bands + slot]
        for i in range(n_bands):
            w[i] = w[width_owner[i]]
        rest = params[n_bands + len(free_widths):]
        d = _design(xf, c, w, shapes, settings, fermi_cache)
        return (d @ rest + fixed - yf) / sigma

    dof = max(int(yf.size - full.size), 1)
    chi2 = float(np.sum(((model - yf) / sigma) ** 2) / dof)
    errors = np.full(full.size, np.nan)
    try:
        jacobian = _numeric_jacobian(full_model, full)
        _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
        keep = singular > (np.finfo(float).eps * max(jacobian.shape) * singular[0])
        covariance = (vt[keep].T / singular[keep] ** 2) @ vt[keep]
        errors = np.sqrt(np.maximum(np.diag(covariance) * chi2, 0.0))
    except (np.linalg.LinAlgError, IndexError, ValueError, FloatingPointError):
        pass

    centre_errors = errors[:n_bands]
    width_errors = np.full(n_bands, np.nan)
    for slot, i in enumerate(free_widths):
        width_errors[i] = errors[n_bands + slot]
    for i in range(n_bands):
        width_errors[i] = width_errors[width_owner[i]]
    height_errors = errors[n_bands + len(free_widths): n_bands + len(free_widths) + n_bands]

    spread = float(np.sum((yf - np.mean(yf)) ** 2))
    r_squared = (1.0 - float(np.sum((yf - model) ** 2)) / spread if spread > 0 else float("nan"))
    areas = np.array([heights[i] * _area_factor(shapes[i], widths[i], settings.resolution)
                      for i in range(n_bands)])

    return LineFit(xf, yf, model, np.asarray(background) * np.ones_like(xf), names, centres,
                   centre_errors, widths, width_errors, heights, height_errors, areas,
                   float(position), chi2, r_squared, shapes, settings.resolution,
                   bool(result.success), str(result.message))


# ==========================================================================
# A whole series
# ==========================================================================
@dataclass
class BandSeries:
    """Every line's fit and the dispersions that come out of them. ``positions`` are the
    fitted lines' coordinates on the across axis; ``along`` is the full fitted axis."""
    positions: np.ndarray
    names: list
    centres: np.ndarray                  # (n_lines, n_bands), NaN where absent
    centre_errors: np.ndarray
    widths: np.ndarray
    width_errors: np.ndarray
    heights: np.ndarray
    areas: np.ndarray
    chi2: np.ndarray
    fits: list = field(default_factory=list)
    direction: str = "mdc"
    along: np.ndarray = None
    along_label: str = ""
    along_unit: str = ""
    across_label: str = ""
    across_unit: str = ""
    settings: FitSettings = None

    def band(self, index: int = 0):
        """``(k, E, k_error)`` for one band in plotting order: for an MDC series the fitted
        quantity is the momentum, for an EDC series the energy."""
        good = np.isfinite(self.centres[:, index])
        if self.direction == "mdc":
            return (self.centres[good, index], self.positions[good], self.centre_errors[good, index])
        return (self.positions[good], self.centres[good, index], self.centre_errors[good, index])

    def error_axis(self) -> str:
        return "x" if self.direction == "mdc" else "y"


def fit_series(values, axes, bands, settings: FitSettings, *, direction: str = "mdc",
               step: int = 1, combine: int = 1, bounds=None, refine: bool = True,
               progress=None, labels=("kx", "Energy"), units=("1/A", "eV")) -> BandSeries:
    """Fit every line of a cut ``values[x, y]`` with axes ``(x, y)`` (momentum, energy).

    ``direction='mdc'`` fits lines along ``x`` at each ``y``; ``'edc'`` the other way round.
    ``bounds`` crops the across axis (the energy window of an MDC series), ``step`` takes every
    n-th line, ``combine`` sums that many neighbouring lines into each fitted one. ``refine``
    runs a second pass seeded from the first. ``progress(done, total)`` may return False to
    stop. ``labels`` / ``units`` name the two axes and are carried into the series.
    """
    values = np.asarray(values, dtype=float)
    x = np.asarray(axes[0], dtype=float)
    y = np.asarray(axes[1], dtype=float)
    if values.shape != (x.size, y.size):
        raise ValueError(f"values{values.shape} does not match axes ({x.size}, {y.size})")
    bands = list(bands.bands) if isinstance(bands, BandModel) else list(bands)
    bands = [b for b in bands if b.seeds]
    if not bands:
        raise ValueError("place at least one seed on one band first")

    if direction == "mdc":
        along, across = x, y
        along_label, along_unit, across_label, across_unit = labels[0], units[0], labels[1], units[1]
        line_of = lambda i: values[:, i]
    elif direction == "edc":
        along, across = y, x
        along_label, along_unit, across_label, across_unit = labels[1], units[1], labels[0], units[0]
        line_of = lambda i: values[i, :]
    else:
        raise ValueError(f"direction {direction!r} is not mdc or edc")

    lo, hi = 0, across.size - 1
    if bounds is not None and bounds != (None, None):
        want_lo = -np.inf if bounds[0] is None else float(bounds[0])
        want_hi = np.inf if bounds[1] is None else float(bounds[1])
        keep = np.flatnonzero((across >= min(want_lo, want_hi)) & (across <= max(want_lo, want_hi)))
        if keep.size == 0:
            raise ValueError("that range contains no lines")
        lo, hi = int(keep[0]), int(keep[-1])

    half = max(0, (int(combine) - 1) // 2)
    indices = list(range(lo, hi + 1, max(1, int(step))))
    tolerance = abs(across[1] - across[0]) if across.size > 1 else 0.0

    def one_pass(previous=None):
        fits = []
        for count, i in enumerate(indices):
            position = float(across[i])
            active = [b for b in bands if b.covers(position, tolerance)]
            if not active:
                fits.append(None)
                continue
            first = max(0, i - half)
            last = min(across.size, i + half + 1)
            line = np.nansum(np.array([line_of(j) for j in range(first, last)]), axis=0)
            guesses = None
            if previous is not None and previous[count] is not None:
                old = previous[count]
                guesses = {name: (old.centres[j], old.widths[j], old.heights[j])
                           for j, name in enumerate(old.names)}
            try:
                fits.append(fit_line(along, line, active, settings, guesses=guesses,
                                     position=position))
            except Exception:
                fits.append(None)
            if progress is not None and not progress(count + 1, len(indices)):
                fits.extend([None] * (len(indices) - len(fits)))
                return fits, False
        return fits, True

    fits, finished = one_pass()
    if refine and finished:
        refined, _ = one_pass(fits)
        fits = [new if new is not None else old for new, old in zip(refined, fits)]

    names = [b.name for b in bands]
    blank = np.full((len(fits), len(names)), np.nan)
    centres, centre_errors = blank.copy(), blank.copy()
    widths, width_errors = blank.copy(), blank.copy()
    heights, areas = blank.copy(), blank.copy()
    chi2 = np.full(len(fits), np.nan)
    for row, fit in enumerate(fits):
        if fit is None:
            continue
        for j, name in enumerate(fit.names):
            column = names.index(name)
            centres[row, column] = fit.centres[j]
            centre_errors[row, column] = fit.centre_errors[j]
            widths[row, column] = fit.widths[j]
            width_errors[row, column] = fit.width_errors[j]
            heights[row, column] = fit.heights[j]
            areas[row, column] = fit.areas[j]
        chi2[row] = fit.chi2

    return BandSeries(np.array([float(across[i]) for i in indices]), names, centres,
                      centre_errors, widths, width_errors, heights, areas, chi2, fits, direction,
                      np.asarray(along, dtype=float), along_label, along_unit, across_label,
                      across_unit, settings)


# ==========================================================================
# Seeding helpers
# ==========================================================================
def suggest_seed(x, y, centre: float, *, window: float = None):
    """Height and FWHM for a seed dropped at ``centre``, read off the line itself."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(abs(x[-1] - x[0])) or 1.0
    window = float(window) if window else span / 8.0
    near = np.abs(x - float(centre)) <= window
    if near.sum() < 3:
        near = np.ones(x.shape, dtype=bool)
    local = y[near]
    floor = float(np.nanmin(local))
    peak = float(np.nanmax(local))
    height = max(peak - floor, 1e-9)
    above = x[near][y[near] - floor >= height / 2.0]
    width = float(above.max() - above.min()) if above.size > 1 else span / 20.0
    return height, max(width, span / 200.0)


def line_at(values, axes, position: float, *, direction: str = "mdc", combine: int = 1):
    """The MDC (or EDC) of ``values[x, y]`` nearest ``position``, summed over ``combine``
    neighbouring lines: ``(actual_position, along_axis, line)``. The seed dialog and the
    explorer's raw-line view use it."""
    values = np.asarray(values, dtype=float)
    x = np.asarray(axes[0], dtype=float)
    y = np.asarray(axes[1], dtype=float)
    along, across = (x, y) if direction == "mdc" else (y, x)
    i = int(np.argmin(np.abs(across - float(position))))
    half = max(0, (int(combine) - 1) // 2)
    first, last = max(0, i - half), min(across.size, i + half + 1)
    block = values[:, first:last] if direction == "mdc" else values[first:last, :].T
    return float(across[i]), along, np.nansum(block, axis=1)


def seeds_from_points(values, axes, points, fwhm: float, *, direction: str = "mdc",
                      combine: int = 5, window: float = None) -> list:
    """Seeds from ``[(position, centre), ...]`` with the height read off the data and a common
    starting FWHM -- the way the prototype's fit scripts built their seeds."""
    out = []
    for position, centre in points:
        actual, along, line = line_at(values, axes, position, direction=direction, combine=combine)
        height, _ = suggest_seed(along, line, centre, window=window)
        out.append(Seed(position=actual, centre=float(centre), fwhm=float(fwhm), height=height))
    return out
