"""Tests for glossy (Phong) shading of atoms, bonds, and polyhedra."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hofmann.model import (
    AtomStyle,
    BondSpec,
    Frame,
    PolyhedronSpec,
    RenderStyle,
    ShadingMode,
    StructureScene,
)
from hofmann.model.composition import Composition
from hofmann.rendering.shading import (
    _Lighting,
    _bond_layer_verts,
    _depth_cue,
    _face_colour,
    _make_bond_shader,
    _make_sphere_shader,
)


def _lighting(**style_kwargs) -> _Lighting:
    return _Lighting.from_style(RenderStyle(**style_kwargs))


def _two_atom_scene(colour_a=(0.2, 0.4, 0.9), colour_b=(0.9, 0.1, 0.1)):
    return StructureScene(
        species=["A", "B"],
        frames=[Frame(coords=np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]))],
        atom_styles={
            "A": AtomStyle(1.0, colour_a),
            "B": AtomStyle(0.8, colour_b),
        },
        bond_specs=[BondSpec(species=("A", "B"), max_length=5.0, radius=0.15)],
    )


def _collection_counts(fig) -> int:
    ax = fig.axes[0]
    return sum(len(c.get_paths()) for c in ax.collections)


# --- RenderStyle fields -------------------------------------------------


class TestShadingStyle:
    def test_default_is_flat(self):
        assert RenderStyle().shading == ShadingMode.FLAT

    def test_string_coerced_to_enum(self):
        assert RenderStyle(shading="glossy").shading is ShadingMode.GLOSSY

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            RenderStyle(shading="shiny")

    @pytest.mark.parametrize("name", [
        "ambient", "diffuse", "specular", "shininess", "polyhedra_specular",
    ])
    def test_negative_lighting_rejected(self, name):
        with pytest.raises(ValueError, match=name):
            RenderStyle(**{name: -0.1})

    @pytest.mark.parametrize("name", [
        "shading_layers", "interactive_shading_layers",
    ])
    def test_layers_must_be_positive(self, name):
        with pytest.raises(ValueError, match=name):
            RenderStyle(**{name: 0})

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_depth_cueing_range(self, value):
        with pytest.raises(ValueError, match="depth_cueing"):
            RenderStyle(depth_cueing=value)

    def test_dict_round_trip(self):
        style = RenderStyle(
            shading="glossy", specular=0.4, shininess=40.0,
            shading_layers=20, depth_cueing=0.3,
        )
        d = style.to_dict()
        assert d["shading"] == "glossy"
        restored = RenderStyle.from_dict(d)
        assert restored.shading is ShadingMode.GLOSSY
        assert restored.specular == 0.4
        assert restored.shininess == 40.0
        assert restored.shading_layers == 20
        assert restored.depth_cueing == 0.3

    def test_defaults_omitted_from_dict(self):
        d = RenderStyle().to_dict()
        assert "shading" not in d
        assert "depth_cueing" not in d

    def test_glossy_preset(self):
        style = RenderStyle.glossy()
        assert style.shading is ShadingMode.GLOSSY
        assert style.atom_outline_width == 0.0
        assert style.bond_outline_width == 0.0
        assert style.depth_cueing > 0.0

    def test_glossy_preset_overrides(self):
        style = RenderStyle.glossy(depth_cueing=0.0, atom_scale=0.3)
        assert style.shading is ShadingMode.GLOSSY
        assert style.depth_cueing == 0.0
        assert style.atom_scale == 0.3

    def test_render_mpl_accepts_shading_kwarg(self):
        fig = _two_atom_scene().render_mpl(shading="glossy", show=False)
        assert fig is not None
        plt.close(fig)


# --- Sphere shader --------------------------------------------------------


class TestSphereShader:
    def test_layer_count_and_outer_layer(self):
        shader = _make_sphere_shader(_lighting(), 16)
        assert len(shader.scales) == 16
        assert len(shader.circles) == 16
        # The outermost layer is the full atom disc.
        assert shader.scales[0] == pytest.approx(1.0)
        np.testing.assert_allclose(shader.offsets[0], [0.0, 0.0])

    def test_layers_shrink_monotonically(self):
        shader = _make_sphere_shader(_lighting(), 24)
        assert np.all(np.diff(shader.scales) < 0)
        assert np.all(shader.scales > 0)

    def test_layers_nested_inside_atom(self):
        # Every layer must lie within the atom disc, including when the
        # highlight is off-centre.
        shader = _make_sphere_shader(
            _lighting(light_direction=(-0.6, 0.7, 1.0)), 24,
        )
        for offset, scale, circle in zip(
            shader.offsets, shader.scales, shader.circles,
        ):
            pts = offset + scale * circle
            assert np.max(np.linalg.norm(pts, axis=1)) <= 1.0 + 1e-9

    def test_headlight_layers_concentric(self):
        shader = _make_sphere_shader(_lighting(), 12)
        np.testing.assert_allclose(shader.offsets, 0.0, atol=1e-12)

    def test_off_axis_light_moves_highlight(self):
        shader = _make_sphere_shader(
            _lighting(light_direction=(-1.0, 1.0, 1.0)), 12,
        )
        # Innermost layer sits towards the upper left.
        assert shader.offsets[-1, 0] < 0
        assert shader.offsets[-1, 1] > 0

    def test_brightens_towards_highlight(self):
        shader = _make_sphere_shader(_lighting(), 24)
        base = (0.4, 0.6, 0.3)
        rim = np.mean(shader.colour(base, 0))
        centre = np.mean(shader.colour(base, len(shader.scales) - 1))
        assert centre > rim
        # The rim is darker than the flat base colour, the centre is
        # whiter than it.
        assert rim < np.mean(base)
        assert centre > np.mean(base)

    def test_default_profile(self):
        # Default lighting: about 45% of the base colour at the
        # silhouette (ambient only), saturating to white at the centre.
        shader = _make_sphere_shader(_lighting(), 48)
        base = (0.44, 0.80, 0.39)
        assert shader.colour(base, len(shader.scales) - 1) == pytest.approx(
            (1.0, 1.0, 1.0), abs=0.02,
        )
        rim = shader.colour(base, 0)
        for channel, b in zip(rim, base):
            assert 0.45 * b <= channel <= 0.75 * b

    def test_matte_has_no_specular(self):
        shader = _make_sphere_shader(_lighting(specular=0.0), 16)
        assert np.all(shader.specs == 0.0)
        base = (0.2, 0.4, 0.6)
        brightest = shader.colour(base, len(shader.scales) - 1)
        for channel, b in zip(brightest, base):
            assert channel <= b + 1e-9

    def test_inner_layers_use_fewer_segments(self):
        shader = _make_sphere_shader(_lighting(), 16, circle_segments=72)
        assert len(shader.circles[0]) == 73
        assert len(shader.circles[-1]) < len(shader.circles[0])


# --- Bond shader ----------------------------------------------------------


class TestBondShader:
    def test_shapes(self):
        bx = np.array([1.0, 0.0])
        by = np.array([0.0, 1.0])
        shader = _make_bond_shader(_lighting(), 8, bx, by)
        assert shader.scales.shape == (8,)
        assert shader.shifts.shape == (8, 2)
        assert shader.factors.shape == (8, 2)

    def test_brighter_on_axis_than_edge(self):
        shader = _make_bond_shader(
            _lighting(), 10, np.array([1.0]), np.array([0.0]),
        )
        base = (0.5, 0.5, 0.5)
        outer = np.mean(shader.colour(base, 0, 0))
        inner = np.mean(shader.colour(base, 9, 0))
        assert inner > outer

    @pytest.mark.parametrize("light", [(0, 0, 1), (-0.7, 0.6, 1.0)])
    def test_layer_polygon_inside_bond(self, light):
        # A horizontal capsule-shaped bond.  Layers keep each vertex's
        # position along the axis and must not widen any cross-section,
        # so every layer stays inside the original outline, caps
        # included, even when the highlight is off-axis.
        arc = np.linspace(0, np.pi, 12)
        start = np.column_stack([-0.3 * np.sin(arc), 0.5 * np.cos(arc)])
        end = np.column_stack([8.0 + 0.3 * np.sin(arc), -0.5 * np.cos(arc)])
        verts = np.concatenate([start, end])
        shader = _make_bond_shader(
            _lighting(light_direction=light), 8,
            np.array([1.0]), np.array([0.0]),
        )
        for i in range(8):
            layer = _bond_layer_verts(
                verts, np.array([0.0, 0.0]), 1.0, 0.0,
                shader.scales[i], shader.shifts[i, 0],
            )
            np.testing.assert_allclose(layer[:, 0], verts[:, 0])
            assert np.all(
                np.abs(layer[:, 1]) <= np.abs(verts[:, 1]) + 1e-12
            )

    def test_layer_preserves_along_axis_coordinate(self):
        verts = np.array([[0.0, 1.0], [5.0, 1.0], [5.0, -1.0], [0.0, -1.0]])
        layer = _bond_layer_verts(
            verts, np.array([0.0, 0.0]), 1.0, 0.0, 0.5, 0.2,
        )
        np.testing.assert_allclose(layer[:, 0], verts[:, 0])
        # Cross-section [-1, 1] maps to [-0.3, 0.7].
        np.testing.assert_allclose(layer[:, 1], [0.7, 0.7, -0.3, -0.3])


# --- Faces and depth cueing ------------------------------------------------


class TestFaceAndFog:
    def test_zero_strength_is_flat(self):
        rgb = (0.3, 0.5, 0.7)
        out = _face_colour(_lighting(), rgb, np.array([0.0, 0.0, 1.0]), 0.0, 1.0)
        assert out == pytest.approx(rgb)

    def test_edge_on_face_darker_than_facing(self):
        rgb = (0.3, 0.5, 0.7)
        lit = _lighting()
        facing = _face_colour(lit, rgb, np.array([0.0, 0.0, 1.0]), 1.0, 0.0)
        edge_on = _face_colour(lit, rgb, np.array([1.0, 0.0, 0.0]), 1.0, 0.0)
        assert np.mean(edge_on) < np.mean(facing)

    def test_face_lighting_two_sided(self):
        rgb = (0.3, 0.5, 0.7)
        lit = _lighting()
        n = np.array([0.3, 0.2, 0.9])
        n /= np.linalg.norm(n)
        assert _face_colour(lit, rgb, n, 1.0, 1.0) == pytest.approx(
            _face_colour(lit, rgb, -n, 1.0, 1.0),
        )

    def test_depth_cue(self):
        assert _depth_cue((0.0, 0.0, 0.0), 0.0, (1, 1, 1)) == (0.0, 0.0, 0.0)
        assert _depth_cue((0.0, 0.5, 1.0), 0.5, (1.0, 1.0, 1.0)) == (
            pytest.approx((0.5, 0.75, 1.0))
        )


# --- Painter integration ----------------------------------------------------


class TestGlossyPainter:
    def _emit_kwargs(self, **overrides):
        defaults = dict(
            centre_xy=np.array([0.0, 0.0]),
            screen_radius=10.0,
            fallback_face_colour=(1.0, 0.0, 0.0, 1.0),
            atom_styles={
                "Fe": AtomStyle(radius=1.0, colour="red"),
                "Mn": AtomStyle(radius=1.0, colour="purple"),
            },
            style=RenderStyle(),
            unit_circle=np.array([
                [1, 0], [0, 1], [-1, 0], [0, -1], [1, 0],
            ], dtype=float),
            show_outlines=True,
            outline_rgb=(0.0, 0.0, 0.0),
            atom_outline_width=1.0,
            use_wedges=True,
            bg_rgb=(1.0, 1.0, 1.0),
            shader=_make_sphere_shader(_lighting(), 6),
        )
        defaults.update(overrides)
        return defaults

    def test_pure_site_layers_then_outline(self):
        from hofmann.rendering.painter import _emit_atom_polygons
        verts, fcs, ecs, lws = _emit_atom_polygons(
            site_content="Fe", **self._emit_kwargs(),
        )
        # Six unstroked layers plus one transparent outline on top.
        assert len(verts) == 7
        assert lws[:6] == [0.0] * 6
        assert fcs[6] == (0.0, 0.0, 0.0, 0.0)
        assert lws[6] == 1.0

    def test_no_outline_when_width_zero(self):
        from hofmann.rendering.painter import _emit_atom_polygons
        verts, fcs, ecs, lws = _emit_atom_polygons(
            site_content="Fe", **self._emit_kwargs(atom_outline_width=0.0),
        )
        assert len(verts) == 6
        assert all(fc[3] == 1.0 for fc in fcs)

    def test_mixed_site_shades_each_wedge(self):
        from hofmann.rendering.painter import _emit_atom_polygons
        verts, fcs, ecs, lws = _emit_atom_polygons(
            site_content=Composition({"Fe": 0.5, "Mn": 0.5}),
            **self._emit_kwargs(show_outlines=False),
        )
        # Two wedges per layer.
        assert len(verts) == 12
        # Each wedge gets brighter towards the centre.
        assert np.mean(fcs[10][:3]) > np.mean(fcs[0][:3])

    def test_unlit_background_vacancy_stays_flat(self):
        from hofmann.rendering.painter import _emit_atom_polygons
        verts, fcs, ecs, lws = _emit_atom_polygons(
            site_content=Composition({"Fe": 0.6}),
            **self._emit_kwargs(show_outlines=False),
        )
        # Layers alternate Fe wedge, vacancy wedge; vacancy = background.
        vacancy_colours = fcs[1::2]
        assert all(fc == (1.0, 1.0, 1.0, 1.0) for fc in vacancy_colours)

    def test_fog_lightens_atom(self):
        from hofmann.rendering.painter import _emit_atom_polygons
        _, near, _, _ = _emit_atom_polygons(
            site_content="Fe", **self._emit_kwargs(),
        )
        _, far, _, _ = _emit_atom_polygons(
            site_content="Fe", **self._emit_kwargs(fog=0.5),
        )
        assert np.mean(far[0][:3]) > np.mean(near[0][:3])

    def test_glossy_render_adds_layers(self):
        scene = _two_atom_scene()
        flat = scene.render_mpl(show=False)
        n_flat = _collection_counts(flat)
        plt.close(flat)
        glossy = scene.render_mpl(show=False, shading="glossy", shading_layers=10)
        n_glossy = _collection_counts(glossy)
        plt.close(glossy)
        # 2 atoms x 10 layers (+2 outlines) and 2 half-bonds x 5 layers
        # (+2 outlines) instead of 2 atoms + 2 half-bonds.
        assert n_flat == 4
        assert n_glossy == 2 * 10 + 2 + 2 * 5 + 2

    def test_glossy_polyhedra_render(self):
        species = ["Ti"] + ["O"] * 6
        coords = np.array([
            [0, 0, 0], [2, 0, 0], [-2, 0, 0],
            [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2],
        ], dtype=float)
        scene = StructureScene(
            species=species,
            frames=[Frame(coords=coords)],
            atom_styles={
                "Ti": AtomStyle(1.0, (0.2, 0.4, 0.9)),
                "O": AtomStyle(0.6, (0.9, 0.1, 0.1)),
            },
            bond_specs=[BondSpec(species=("Ti", "O"), max_length=2.5)],
            polyhedra=[PolyhedronSpec(centre="Ti")],
        )
        fig = scene.render_mpl(show=False, style=RenderStyle.glossy())
        plt.close(fig)

    def test_flat_unchanged_by_lighting_fields(self):
        # Lighting coefficients have no effect in flat mode.
        scene = _two_atom_scene()
        a = scene.render_mpl(show=False)
        b = scene.render_mpl(show=False, specular=0.0, ambient=0.1)
        fa = np.concatenate([c.get_facecolors() for c in a.axes[0].collections])
        fb = np.concatenate([c.get_facecolors() for c in b.axes[0].collections])
        np.testing.assert_array_equal(fa, fb)
        plt.close(a)
        plt.close(b)

    def test_depth_cueing_fades_back_atom(self):
        scene = StructureScene(
            species=["A", "A"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, -3.0]]))],
            atom_styles={"A": AtomStyle(1.0, (0.0, 0.0, 0.0))},
        )
        fig = scene.render_mpl(
            show=False, depth_cueing=0.5, show_outlines=False,
        )
        fcs = fig.axes[0].collections[0].get_facecolors()
        plt.close(fig)
        # Painter order: back atom first (fogged half-way to white),
        # front atom last (unchanged).
        np.testing.assert_allclose(fcs[0][:3], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(fcs[-1][:3], [0.0, 0.0, 0.0])

    def test_glossy_legend_renders(self):
        fig = _two_atom_scene().render_mpl(
            show=False, style=RenderStyle.glossy(show_legend=True),
        )
        plt.close(fig)
