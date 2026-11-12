"""Tests for the ``hofmann`` command-line interface."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

from hofmann.cli import (  # noqa: E402
    _view_direction,
    build_parser,
    find_structure_files,
    load_colour_overrides,
    main,
    parse_inputs,
    read_structure,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    shutil.copy(EXAMPLES / "srtio3.cif", tmp_path / "srtio3.cif")
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestInputs:
    def test_views_elements_and_files(self, workdir):
        parsed = parse_inputs(
            ["x", "Ti", "O", "srtio3.cif"], build_parser(),
        )
        assert parsed.views == ["x"]
        assert parsed.poly_pairs == [("Ti", "O")]
        assert parsed.paths == [Path("srtio3.cif")]

    def test_any_order(self, workdir):
        parsed = parse_inputs(
            ["srtio3.cif", "Ti", "O", "zx", "Sr", "O"], build_parser(),
        )
        assert parsed.views == ["z", "x"]
        assert parsed.poly_pairs == [("Ti", "O"), ("Sr", "O")]
        assert parsed.paths == [Path("srtio3.cif")]

    def test_format_words(self, workdir):
        parsed = parse_inputs(
            ["x", "P", "S", "svg", "PDF", "srtio3.cif"], build_parser(),
        )
        assert parsed.formats == ["svg", "pdf"]
        assert parsed.poly_pairs == [("P", "S")]
        assert parsed.paths == [Path("srtio3.cif")]

    def test_view_letters_deduplicated(self, workdir):
        parsed = parse_inputs(["wx", "xw", "y"], build_parser())
        assert parsed.views == ["w", "x", "y"]

    def test_no_views_given(self, workdir):
        assert parse_inputs(["srtio3.cif"], build_parser()).views == []

    def test_odd_elements_rejected(self, workdir):
        with pytest.raises(SystemExit):
            parse_inputs(["srtio3.cif", "Ti"], build_parser())

    @pytest.mark.parametrize("token", ["missing.cif", "xq", "X", "ti"])
    def test_unknown_token_rejected(self, workdir, token):
        with pytest.raises(SystemExit):
            parse_inputs([token], build_parser())

    def test_find_structure_files(self, tmp_path):
        for name in [
            "a.cif", "b.vasp", "c.poscar", "POSCAR", "CONTCAR_relaxed",
            "POSCAR.png", "notes.txt", "a.png",
        ]:
            (tmp_path / name).write_text("x")
        found = [p.name for p in find_structure_files(tmp_path)]
        assert found == sorted(
            ["a.cif", "b.vasp", "c.poscar", "POSCAR", "CONTCAR_relaxed"]
        )


class TestReadStructure:
    def test_poscar_without_extension(self, tmp_path):
        s = read_structure(EXAMPLES / "srtio3.cif")
        s.to(filename=str(tmp_path / "POSCAR"), fmt="poscar")
        assert read_structure(tmp_path / "POSCAR").composition == s.composition

    def test_vasp_extension(self):
        s = read_structure(EXAMPLES / "Zr4S12N8H24.vasp")
        assert len(s) == 48


class TestColours:
    def test_yaml_forms(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text(
            'Li: "#ff0000"\n'
            "S: [0.0, 0.0, 1.0]\n"
            "La: {colour: white, radius: 1.5}\n"
        )
        colours = load_colour_overrides(f)
        assert colours["Li"]["colour"] == (1.0, 0.0, 0.0)
        assert colours["S"]["colour"] == (0.0, 0.0, 1.0)
        assert colours["La"] == {"colour": (1.0, 1.0, 1.0), "radius": 1.5}

    def test_json(self, tmp_path):
        f = tmp_path / "c.json"
        f.write_text('{"O": "black"}')
        assert load_colour_overrides(f)["O"]["colour"] == (0.0, 0.0, 0.0)

    def test_not_a_mapping(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text("- Li\n- S\n")
        with pytest.raises(ValueError):
            load_colour_overrides(f)


class TestView:
    @pytest.mark.parametrize("view", ["x", "y", "z", "w", "1,1,0", "1 1 1"])
    def test_valid(self, view):
        s = read_structure(EXAMPLES / "srtio3.cif")
        direction, up = _view_direction(s, view)
        assert direction.shape == (3,)
        assert up.shape == (3,)

    @pytest.mark.parametrize("view, expected", [
        ("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1]),
    ])
    def test_axis_views(self, view, expected):
        s = read_structure(EXAMPLES / "srtio3.cif")
        direction, _ = _view_direction(s, view)
        assert list(direction) == pytest.approx(expected)

    @pytest.mark.parametrize("view", ["a", "1,2", "0,0,0"])
    def test_invalid(self, view):
        s = read_structure(EXAMPLES / "srtio3.cif")
        with pytest.raises(ValueError):
            _view_direction(s, view)


class TestMain:
    def test_dry_run_scans_directory(self, workdir, capsys):
        assert main(["--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "Ti-O: polyhedra" in out
        assert "Sr-O: none" in out
        assert not list(workdir.glob("*.png"))

    def test_element_shorthand_overrides(self, workdir, capsys):
        assert main(["srtio3.cif", "Sr", "O", "-n"]) == 0
        assert "Sr-O: polyhedra" in capsys.readouterr().out

    def test_nobond(self, workdir, capsys):
        assert main(["srtio3.cif", "--nobond", "Ti", "O", "-n"]) == 0
        assert "Ti-O: none" in capsys.readouterr().out

    def test_no_auto(self, workdir, capsys):
        assert main(["srtio3.cif", "--no-auto", "-n"]) == 0
        out = capsys.readouterr().out
        assert "polyhedra" not in out

    def test_renders_four_views_by_default(self, workdir):
        assert main(["--dpi", "40", "--size", "2"]) == 0
        for view in "xyzw":
            assert (workdir / f"srtio3_{view}.png").stat().st_size > 0

    def test_single_view(self, workdir):
        assert main(["x", "Ti", "O", "srtio3.cif", "--dpi", "40",
                     "--size", "2"]) == 0
        assert sorted(p.name for p in workdir.glob("*.png")) == [
            "srtio3_x.png",
        ]

    def test_positional_formats(self, workdir):
        assert main(["x", "svg", "pdf", "srtio3.cif", "--flat",
                     "--size", "2"]) == 0
        assert sorted(p.name for p in workdir.iterdir()
                      if p.suffix != ".cif") == [
            "srtio3_x.pdf", "srtio3_x.svg",
        ]

    def test_format_flag_and_word_combined(self, workdir, capsys):
        assert main(["z", "svg", "srtio3.cif", "-f", "pdf", "-n"]) == 0
        out = capsys.readouterr().out
        assert "srtio3_z.svg" in out
        assert "srtio3_z.pdf" in out
        assert "srtio3_z.png" not in out

    def test_custom_direction(self, workdir, capsys):
        assert main(["srtio3.cif", "--view", "1,1,0", "-n"]) == 0
        out = capsys.readouterr().out
        assert "srtio3_1-1-0.png" in out
        assert "srtio3_x.png" not in out

    def test_bad_custom_direction(self, workdir):
        with pytest.raises(SystemExit):
            main(["srtio3.cif", "--view", "1,1", "-n"])

    def test_outdir_and_format(self, workdir):
        out = workdir / "figs"
        assert main(
            ["z", "srtio3.cif", "-o", str(out), "-f", "svg", "--flat",
             "--size", "2"],
        ) == 0
        assert (out / "srtio3_z.svg").is_file()

    def test_colour_file_picked_up(self, workdir, monkeypatch):
        (workdir / "hofmann_colours.yaml").write_text("Ti: black\n")
        seen = {}
        import hofmann.cli as cli

        original = cli.render_file

        def spy(path, outputs, args, poly_pairs, colours):
            seen.update(colours)
            return original(path, outputs, args, poly_pairs, colours)

        monkeypatch.setattr(cli, "render_file", spy)
        assert main(["-n"]) == 0
        assert seen["Ti"]["colour"] == (0.0, 0.0, 0.0)

    def test_bad_file_reported_and_others_continue(self, workdir, capsys):
        (workdir / "broken.cif").write_text("not a cif\n")
        assert main(["-n"]) == 1
        captured = capsys.readouterr()
        assert "broken.cif: error" in captured.err
        assert "srtio3.cif" in captured.out

    def test_empty_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert main([]) == 1
