"""Glossy (Phong) shading for atoms, bonds, and polyhedron faces.

matplotlib has no per-pixel lighting, so a smoothly shaded sphere is
approximated by a stack of nested, filled polygons.  Each layer is an
affine copy of the atom's flat outline, shrunk towards the screen
position of the specular highlight and filled with the Phong colour
of the band it exposes.  The layers stay plain vector paths, so
glossy output remains resolution-independent in SVG and PDF.

Bonds use the same idea in one dimension: each layer narrows the
bond polygon across its width, towards the cylinder's highlight line.

All lighting is evaluated in screen space (x right, y up, z towards
the viewer) with an orthographic viewer along +z, using

    colour = base * (ambient + diffuse * max(n.L, 0))
             + specular * max(n.H, 0) ** shininess

clamped to ``[0, 1]``, where *L* is the light direction and *H* the
Blinn half-vector between *L* and the view direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fine grid used to measure how fast the shaded colour changes when
# choosing where to place layer boundaries.
_PROFILE_SAMPLES = 256
# Points sampled around each layer boundary when averaging the colour
# of a band (spheres only; bonds are sampled exactly).
_RING_SAMPLES = 48


@dataclass(frozen=True)
class _Lighting:
    """Phong lighting coefficients, resolved from a :class:`RenderStyle`."""

    light: np.ndarray
    half: np.ndarray
    ambient: float
    diffuse: float
    specular: float
    shininess: float

    @classmethod
    def from_style(cls, style) -> _Lighting:
        light = np.asarray(style.light_direction, dtype=float)
        light = light / np.linalg.norm(light)
        half = light + np.array([0.0, 0.0, 1.0])
        norm = np.linalg.norm(half)
        half = half / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        return cls(
            light=light,
            half=half,
            ambient=float(style.ambient),
            diffuse=float(style.diffuse),
            specular=float(style.specular),
            shininess=float(style.shininess),
        )

    def terms(self, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(diffuse_factor, specular_add)`` for unit *normals*."""
        n_dot_l = np.clip(normals @ self.light, 0.0, None)
        n_dot_h = np.clip(normals @ self.half, 0.0, None)
        factor = self.ambient + self.diffuse * n_dot_l
        spec = self.specular * n_dot_h ** self.shininess
        return factor, spec


def _shade_rgb(
    rgb: tuple[float, ...] | np.ndarray, factor: float, spec: float,
) -> tuple[float, float, float]:
    """Apply a diffuse factor and additive specular term to *rgb*."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    return (
        min(1.0, max(0.0, r * factor + spec)),
        min(1.0, max(0.0, g * factor + spec)),
        min(1.0, max(0.0, b * factor + spec)),
    )


def _layer_params(profile: np.ndarray, n_layers: int) -> np.ndarray:
    """Choose *n_layers* shrink parameters ``t`` in ``[0, 1)``.

    Layers are placed where the colour changes fastest, so that each
    step between adjacent bands is roughly equal in brightness.  A
    uniform term keeps layers from collapsing entirely in flat regions.

    Args:
        profile: ``(_PROFILE_SAMPLES, 2)`` array of ``(factor, spec)``
            along ``t = linspace(0, 1, _PROFILE_SAMPLES)``.
        n_layers: Number of layers (``>= 1``).
    """
    t_grid = np.linspace(0.0, 1.0, len(profile))
    brightness = 0.6 * profile[:, 0] + profile[:, 1]
    change = np.abs(np.diff(brightness))
    total = change.sum()
    weight = change + (total / len(change) * 0.35 if total > 0 else 1.0)
    cumulative = np.concatenate([[0.0], np.cumsum(weight)])
    cumulative /= cumulative[-1]
    targets = np.arange(n_layers) / n_layers
    return np.interp(targets, cumulative, t_grid)


@dataclass(frozen=True)
class _SphereShader:
    """Layer geometry and lighting terms shared by every atom in a draw.

    Layer *i* maps a unit-radius outline point ``p`` to
    ``offsets[i] + scales[i] * p``; multiply by the atom's screen
    radius and add its centre to get screen coordinates.
    """

    offsets: np.ndarray  # (n_layers, 2)
    scales: np.ndarray   # (n_layers,)
    factors: np.ndarray  # (n_layers,)
    specs: np.ndarray    # (n_layers,)
    circles: tuple[np.ndarray, ...]
    """Unit circle for each layer.  Inner layers are smaller on screen
    and use proportionally fewer segments, which keeps vector output
    compact without visibly faceting the outline."""

    def colour(self, rgb, i: int) -> tuple[float, float, float]:
        return _shade_rgb(rgb, self.factors[i], self.specs[i])


def _sphere_normals(points_2d: np.ndarray) -> np.ndarray:
    """Front-hemisphere unit normals for points inside the unit disc."""
    r2 = np.sum(points_2d**2, axis=-1)
    z = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    normals = np.concatenate([points_2d, z[..., None]], axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def _layer_circle(n: int) -> np.ndarray:
    """Closed unit circle with *n* segments (same layout as
    ``projection._make_unit_circle``, which cannot be imported here
    without a cycle)."""
    angles = np.linspace(0.0, 2.0 * np.pi, n + 1)
    return np.column_stack([np.cos(angles), np.sin(angles)])


# Fewest segments used for the circle of any shading layer.
_MIN_LAYER_SEGMENTS = 12


def _make_sphere_shader(
    lighting: _Lighting, n_layers: int, circle_segments: int = 72,
) -> _SphereShader:
    """Build the nested-layer description of a lit unit sphere.

    Args:
        lighting: Resolved lighting coefficients.
        n_layers: Number of nested layers.
        circle_segments: Segment count of the outermost layer's
            circle (the atom outline).
    """
    # Layers shrink towards the screen position of the specular peak.
    focus = lighting.half[:2].copy()
    focus_r = np.linalg.norm(focus)
    if focus_r > 0.95:
        focus *= 0.95 / focus_r

    angles = np.linspace(0.0, 2.0 * np.pi, _RING_SAMPLES, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles)])

    def band_terms(t: np.ndarray) -> np.ndarray:
        # Mean (factor, spec) around the circle of layer t.
        pts = t[:, None, None] * focus + (1.0 - t)[:, None, None] * ring
        factor, spec = lighting.terms(_sphere_normals(pts))
        return np.column_stack([factor.mean(axis=1), spec.mean(axis=1)])

    t_grid = np.linspace(0.0, 1.0, _PROFILE_SAMPLES)
    t_layers = _layer_params(band_terms(t_grid), n_layers)
    # Colour each band by the terms midway to the next layer boundary.
    t_next = np.append(t_layers[1:], 1.0)
    terms = band_terms(0.5 * (t_layers + t_next))
    scales = 1.0 - t_layers
    cache: dict[int, np.ndarray] = {}
    circles = []
    for i, scale in enumerate(scales):
        n = circle_segments if i == 0 else min(
            circle_segments,
            max(_MIN_LAYER_SEGMENTS, int(np.ceil(circle_segments * scale))),
        )
        if n not in cache:
            cache[n] = _layer_circle(n)
        circles.append(cache[n])
    return _SphereShader(
        offsets=t_layers[:, None] * focus,
        scales=scales,
        factors=terms[:, 0],
        specs=terms[:, 1],
        circles=tuple(circles),
    )


@dataclass(frozen=True)
class _BondShader:
    """Layer parameters for shading bond cylinders across their width.

    Layer *i* maps a vertex of bond *b* at signed perpendicular
    offset ``w`` from the bond axis to
    ``scales[i] * w + shifts[i, b] * |w|``.  Because every
    cross-section of a bond polygon is symmetric about the axis, this
    narrows each cross-section to a sub-interval of itself, centred
    on the bond's highlight line.
    """

    scales: np.ndarray   # (n_layers,)
    shifts: np.ndarray   # (n_layers, n_bonds)
    factors: np.ndarray  # (n_layers, n_bonds)
    specs: np.ndarray    # (n_layers, n_bonds)

    def colour(self, rgb, i: int, bond: int) -> tuple[float, float, float]:
        return _shade_rgb(rgb, self.factors[i, bond], self.specs[i, bond])


def _cylinder_terms(
    lighting: _Lighting, s: np.ndarray, perp: np.ndarray,
) -> np.ndarray:
    """``(factor, spec)`` at signed width fraction *s* across bonds.

    Args:
        s: ``(..., n_bonds)`` positions across the width in ``[-1, 1]``.
        perp: ``(n_bonds, 2)`` unit screen perpendicular of each bond.
    """
    z = np.sqrt(np.clip(1.0 - s**2, 0.0, None))
    normals = np.stack(
        [s * perp[:, 0], s * perp[:, 1], z], axis=-1,
    )
    factor, spec = lighting.terms(normals)
    return np.stack([factor, spec], axis=-1)


def _make_bond_shader(
    lighting: _Lighting, n_layers: int, bx: np.ndarray, by: np.ndarray,
) -> _BondShader:
    """Build per-bond layer colours for cylinders along ``(bx, by)``.

    The layer geometry (where the bands sit) is shared by all bonds
    and placed using a horizontal reference bond; only the colours
    are evaluated per bond, from the exact cylinder normals at each
    band's centre.
    """
    perp = np.column_stack([-by, bx]) if len(bx) else np.empty((0, 2))
    # Highlight line: where the cylinder normal best faces H.
    h_ref = lighting.half[:2]

    def ref_profile(t: np.ndarray) -> np.ndarray:
        # Brightness profile for a bond running along screen x,
        # averaged over both edges of the band.
        focus = h_ref[1] * 0.9
        lo = t * focus - (1.0 - t)
        hi = t * focus + (1.0 - t)
        ref = np.array([[0.0, 1.0]])
        a = _cylinder_terms(lighting, lo[:, None], ref)[:, 0]
        b = _cylinder_terms(lighting, hi[:, None], ref)[:, 0]
        return 0.5 * (a + b)

    t_grid = np.linspace(0.0, 1.0, _PROFILE_SAMPLES)
    t_layers = _layer_params(ref_profile(t_grid), n_layers)
    t_next = np.append(t_layers[1:], 1.0)
    t_mid = 0.5 * (t_layers + t_next)

    # Per-bond highlight position across the width.
    focus_b = np.clip(perp @ h_ref, -0.9, 0.9) if len(bx) else np.empty(0)
    # Band centre colour: the two edges of the band at t_mid.
    lo = t_mid[:, None] * focus_b[None, :] - (1.0 - t_mid)[:, None]
    hi = t_mid[:, None] * focus_b[None, :] + (1.0 - t_mid)[:, None]
    terms = 0.5 * (
        _cylinder_terms(lighting, lo, perp)
        + _cylinder_terms(lighting, hi, perp)
    )
    return _BondShader(
        scales=1.0 - t_layers,
        shifts=t_layers[:, None] * focus_b[None, :],
        factors=terms[..., 0],
        specs=terms[..., 1],
    )


def _bond_layer_verts(
    verts: np.ndarray,
    origin: np.ndarray,
    bx: float,
    by: float,
    scale: float,
    shift: float,
) -> np.ndarray:
    """Narrow one bond polygon across its width for a shading layer.

    Args:
        verts: ``(n, 2)`` polygon vertices in screen space.
        origin: A point on the bond axis.
        bx, by: Unit screen direction of the bond.
        scale: Width scale ``1 - t`` of the layer.
        shift: Offset of the layer centre as a fraction of the local
            half-width (``t * focus``).
    """
    rel = verts - origin
    perp = np.array([-by, bx])
    w = rel @ perp
    w_new = scale * w + shift * np.abs(w)
    return verts + np.outer(w_new - w, perp)


def _face_colour(
    lighting: _Lighting,
    rgb: tuple[float, ...],
    normal: np.ndarray,
    strength: float,
    specular_scale: float,
) -> tuple[float, float, float]:
    """Phong colour of a flat polyhedron face (two-sided lighting).

    Args:
        normal: Unit face normal in screen space (either orientation).
        strength: Blend between the flat base colour (``0``) and the
            fully lit colour (``1``).
        specular_scale: Multiplier on the specular coefficient for
            faces.
    """
    n = normal if normal[2] >= 0 else -normal
    n_dot_l = abs(float(n @ lighting.light))
    n_dot_h = abs(float(n @ lighting.half))
    factor = lighting.ambient + lighting.diffuse * n_dot_l
    spec = lighting.specular * specular_scale * n_dot_h ** lighting.shininess
    factor = 1.0 + strength * (factor - 1.0)
    spec *= strength
    return _shade_rgb(rgb, factor, spec)


def _depth_cue(
    rgb: tuple[float, ...],
    amount: float,
    bg_rgb: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Blend *rgb* towards the background by *amount* in ``[0, 1]``."""
    if amount <= 0.0:
        return (rgb[0], rgb[1], rgb[2])
    keep = 1.0 - amount
    return (
        rgb[0] * keep + bg_rgb[0] * amount,
        rgb[1] * keep + bg_rgb[1] * amount,
        rgb[2] * keep + bg_rgb[2] * amount,
    )
