"""Command-line interface: render structure files to images.

Positional arguments can be given in any order and are told apart by
their form:

* a word made only of the letters ``w``, ``x``, ``y``, ``z`` picks the
  view directions: ``x`` = [100], ``y`` = [010], ``z`` = [001],
  ``w`` = an oblique view.  The default is all four (``xyzw``);
* an element symbol (capitalised: ``P``, ``Li``) -- consecutive pairs
  are drawn as polyhedra, like ``--poly``;
* ``png``, ``svg`` or ``pdf`` -- the image format(s), default ``png``;
* anything else must be an existing file or directory.

Examples::

    hofmann                       # every structure file here, 4 views
    hofmann Li3PS4.cif            # one file, 4 views
    hofmann x P S Li3PS4.cif      # P-S polyhedra, [100] view only
    hofmann x P S svg Li3PS4.cif  # the same, as SVG
    hofmann xz Li3PS4.cif --nobond Li S
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hofmann.model.colour import normalise_colour

#: Filename patterns picked up when scanning a directory.
STRUCTURE_PATTERNS: tuple[str, ...] = (
    "*.cif", "*.vasp", "*.poscar", "POSCAR*", "CONTCAR*",
)
_IMAGE_SUFFIXES = frozenset({
    ".png", ".svg", ".pdf", ".jpg", ".jpeg", ".gif", ".mp4",
})
#: Colour files picked up automatically, first match wins per location.
COLOUR_FILE_NAMES: tuple[str, ...] = (
    "hofmann_colours.yaml", "hofmann_colours.yml", "hofmann_colours.json",
)
_USER_COLOUR_FILE = Path("~/.config/hofmann/colours.yaml")

#: View letters and what they look along, in default output order.
VIEWS: dict[str, str] = {
    "x": "[100]",
    "y": "[010]",
    "z": "[001]",
    "w": "oblique",
}
_VIEW_WORD = re.compile(r"^[wxyz]+$")
#: Image formats accepted as positional words or with ``-f``.
FORMATS: tuple[str, ...] = ("png", "svg", "pdf")
# Oblique view: Cartesian weights of the unit a, b, c axes for the
# direction towards the viewer, chosen to show all three cell axes.
_OBLIQUE_WEIGHTS = (1.0, 0.55, 0.4)


def _is_element(token: str) -> bool:
    from pymatgen.core.periodic_table import Element

    return Element.is_valid_symbol(token)


def _is_structure_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() in _IMAGE_SUFFIXES:
        return False
    return any(
        fnmatch.fnmatchcase(path.name, pat)
        or fnmatch.fnmatchcase(path.name.lower(), pat.lower())
        for pat in STRUCTURE_PATTERNS
    )


def find_structure_files(directory: Path) -> list[Path]:
    """Structure files directly inside *directory*, sorted by name."""
    return sorted(p for p in directory.iterdir() if _is_structure_file(p))


def read_structure(path: Path):
    """Read a CIF or VASP POSCAR/CONTCAR file into a pymatgen Structure."""
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar

    name = path.name.lower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if (
            path.suffix.lower() in (".vasp", ".poscar")
            or "poscar" in name or "contcar" in name
        ):
            return Poscar.from_file(str(path)).structure
        return Structure.from_file(str(path))


def _load_mapping(path: Path) -> dict:
    """Load a YAML or JSON mapping."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(text)
    else:
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            from ruamel.yaml import YAML

            data = YAML(typ="safe").load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of element -> colour")
    return data


def load_colour_overrides(path: Path) -> dict[str, dict]:
    """Read element colour (and optional radius) overrides.

    The file maps element symbols to a colour, or to a mapping with
    ``colour`` and/or ``radius`` keys.  Colours accept any format
    understood by :func:`~hofmann.normalise_colour`::

        Li: "#7bc86c"
        S: [1.0, 0.85, 0.2]
        La: {colour: seagreen, radius: 1.4}

    Returns:
        Element -> ``{"colour": rgb, "radius": float}`` (keys present
        only when given).
    """
    out: dict[str, dict] = {}
    for element, value in _load_mapping(path).items():
        entry: dict = {}
        if isinstance(value, dict):
            if "colour" in value or "color" in value:
                entry["colour"] = normalise_colour(
                    value.get("colour", value.get("color")),
                )
            if "radius" in value:
                entry["radius"] = float(value["radius"])
        else:
            if isinstance(value, list):
                value = tuple(value)
            entry["colour"] = normalise_colour(value)
        out[str(element)] = entry
    return out


def _default_colour_files(cwd: Path) -> list[Path]:
    files: list[Path] = []
    user = _USER_COLOUR_FILE.expanduser()
    if user.is_file():
        files.append(user)
    for name in COLOUR_FILE_NAMES:
        candidate = cwd / name
        if candidate.is_file():
            files.append(candidate)
            break
    return files


def _parse_direction(text: str) -> np.ndarray:
    """Parse a lattice direction ``'u,v,w'`` (or ``'u v w'``)."""
    try:
        uvw = np.array([float(x) for x in text.replace(",", " ").split()])
    except ValueError:
        uvw = np.empty(0)
    if uvw.shape != (3,) or not np.any(uvw):
        raise ValueError(
            f"a view must be letters from 'wxyz' or a lattice direction "
            f"'u,v,w', got {text!r}"
        )
    return uvw


def _view_direction(structure, view: str) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian viewing direction and up vector for a view.

    Args:
        structure: The pymatgen structure (for its lattice).
        view: ``"x"``, ``"y"``, ``"z"``, ``"w"``, or a lattice
            direction ``"u,v,w"``.
    """
    lattice = structure.lattice.matrix
    a_hat, b_hat, c_hat = (row / np.linalg.norm(row) for row in lattice)
    if view == "x":
        return a_hat, c_hat
    if view == "y":
        return b_hat, c_hat
    if view == "z":
        return c_hat, b_hat
    if view == "w":
        wa, wb, wc = _OBLIQUE_WEIGHTS
        return wa * a_hat + wb * b_hat + wc * c_hat, c_hat
    direction = _parse_direction(view) @ lattice
    unit = direction / np.linalg.norm(direction)
    up = c_hat if abs(float(unit @ c_hat)) < 0.9 else b_hat
    return direction, up


def _view_suffix(view: str) -> str:
    if view in VIEWS:
        return view
    return "-".join(f"{x:g}" for x in _parse_direction(view))


def _pairs(values: list[list[str]] | None) -> list[tuple[str, str]]:
    return [(a, b) for a, b in (values or [])]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hofmann",
        usage="hofmann [VIEWS] [A B ...] [FORMAT ...] [FILE|DIR ...] [options]",
        description=(
            "Render crystal structure files (CIF, POSCAR/CONTCAR, .vasp) "
            "to images.  With no files, renders every structure file in "
            "the current directory.\n\n"
            "Positional arguments may come in any order:\n"
            "  VIEWS     letters from 'wxyz': x=[100] y=[010] z=[001] "
            "w=oblique\n"
            "            (default: all four, one image each)\n"
            "  A B       element pairs, drawn as polyhedra (like --poly)\n"
            "  FORMAT    png, svg and/or pdf (default: png)\n"
            "  FILE|DIR  structure files or directories\n\n"
            "For ionic compounds, bonds and polyhedra are chosen "
            "automatically from\nestimated bond strengths (|q1*q2|/d): "
            "the most strongly bonded cations\nget polyhedra, the weakest "
            "are left unbonded, and like charges never\nbond.  Metals and "
            "alloys show all nearest-neighbour bonds."
        ),
        epilog=(
            "examples:\n"
            "  hofmann                         all files here, 4 views\n"
            "  hofmann Li3PS4.cif              one file, 4 views\n"
            "  hofmann x P S Li3PS4.cif        P-S polyhedra, [100] only\n"
            "  hofmann x P S svg Li3PS4.cif    the same, as SVG\n"
            "  hofmann xw Li3PS4.cif --nobond Li S\n"
            "  hofmann LLZO.cif --bond Zr O --view 1,1,0 -f pdf\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument(
        "--poly", nargs=2, action="append", metavar=("A", "B"),
        help="draw A-B bonds as polyhedra around the cation (repeatable)",
    )
    parser.add_argument(
        "--bond", nargs=2, action="append", metavar=("A", "B"),
        help="draw A-B bonds as sticks (repeatable)",
    )
    parser.add_argument(
        "--nobond", nargs=2, action="append", metavar=("A", "B"),
        help="do not draw A-B bonds (repeatable)",
    )
    parser.add_argument(
        "--no-auto", action="store_true",
        help=(
            "skip the automatic bond-strength analysis; draw all default "
            "bonds, then apply --poly/--bond/--nobond"
        ),
    )
    parser.add_argument(
        "--view", action="append", metavar="U,V,W",
        help=(
            "extra view along lattice direction u,v,w (or wxyz letters); "
            "repeatable"
        ),
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, default=None,
        help="output directory (default: next to each input file)",
    )
    parser.add_argument(
        "-f", "--format", action="append", choices=FORMATS,
        dest="formats",
        help="image format (repeatable; default: png)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="resolution (default: 300)",
    )
    parser.add_argument(
        "--size", type=float, default=6.0,
        help="figure size in inches (default: 6)",
    )
    parser.add_argument(
        "--supercell", nargs=3, type=int, metavar=("NA", "NB", "NC"),
        help="build a supercell before rendering",
    )
    parser.add_argument(
        "--flat", action="store_true",
        help="flat XBS-style shading instead of glossy",
    )
    parser.add_argument(
        "--colours", "--colors", type=Path, default=None,
        dest="colours",
        help=(
            "YAML/JSON file of element colours (default: "
            "~/.config/hofmann/colours.yaml and ./hofmann_colours.yaml "
            "if present)"
        ),
    )
    parser.add_argument(
        "--no-cell", action="store_true", help="hide the unit cell",
    )
    parser.add_argument(
        "--no-axes", action="store_true", help="hide the axes widget",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="print the chosen bonds, polyhedra, and outputs only",
    )
    return parser


@dataclass
class ParsedInputs:
    """Positional arguments sorted by kind."""

    paths: list[Path] = field(default_factory=list)
    views: list[str] = field(default_factory=list)
    poly_pairs: list[tuple[str, str]] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)


def parse_inputs(
    tokens: Sequence[str], parser: argparse.ArgumentParser,
) -> ParsedInputs:
    """Sort positional tokens into views, formats, element pairs, paths.

    A token made only of the letters ``wxyz`` is a view word;
    ``png``/``svg``/``pdf`` is an image format; an existing path is a
    file or directory; a valid element symbol is an element.  Element
    symbols are taken in consecutive pairs.
    """
    parsed = ParsedInputs()
    elements: list[str] = []
    for token in tokens:
        if _VIEW_WORD.match(token):
            for letter in token:
                if letter not in parsed.views:
                    parsed.views.append(letter)
        elif token.lower() in FORMATS:
            if token.lower() not in parsed.formats:
                parsed.formats.append(token.lower())
        elif Path(token).exists():
            parsed.paths.append(Path(token))
        elif _is_element(token):
            elements.append(token)
        else:
            parser.error(
                f"{token!r} is not a view (letters wxyz), an element, an "
                "image format (png/svg/pdf), or an existing file"
            )
    if len(elements) % 2:
        parser.error(
            "elements must come in pairs, e.g. 'hofmann x P S file.cif', "
            f"got {' '.join(elements)}"
        )
    parsed.poly_pairs = list(zip(elements[0::2], elements[1::2]))
    return parsed


def _output_path(
    src: Path, outdir: Path | None, view: str, fmt: str, used: set[Path],
) -> Path:
    directory = outdir if outdir is not None else src.parent
    stem = (
        src.stem if src.suffix.lower() in (".cif", ".vasp", ".poscar")
        else src.name
    )
    out = directory / f"{stem}_{_view_suffix(view)}.{fmt}"
    if out in used:
        out = directory / (
            f"{src.name.replace('.', '_')}_{_view_suffix(view)}.{fmt}"
        )
    used.add(out)
    return out


def render_file(
    path: Path,
    outputs: dict[tuple[str, str], Path],
    args: argparse.Namespace,
    poly_pairs: list[tuple[str, str]],
    colours: dict[str, dict],
) -> list[str]:
    """Analyse one structure file and render it once per view.

    Args:
        path: Structure file.
        outputs: ``(view, format)`` -> output image path, in render
            order.
        args: Parsed command-line options.
        poly_pairs: Element pairs to draw as polyhedra.
        colours: Element colour / radius overrides.

    Returns:
        Summary lines describing the formula, the chosen scheme, and
        the images written.
    """
    from hofmann.construction.auto_bonding import (
        BondingScheme,
        PairDisplay,
        PairRule,
        analyse_bonding,
        apply_overrides,
    )
    from hofmann.construction.defaults import default_bond_specs
    from hofmann.construction.scene_builders import from_pymatgen
    from hofmann.model import RenderStyle

    structure = read_structure(path)
    if args.supercell:
        structure.make_supercell(args.supercell)

    if args.no_auto:
        elements = sorted(
            el.symbol
            for el in structure.composition.element_composition.elements
        )
        scheme = BondingScheme(oxidation_states={}, metallic=True)
        for spec in default_bond_specs(elements):
            scheme.rules[spec.species] = PairRule(
                species=spec.species, display=PairDisplay.BOND,
                max_length=spec.max_length,
            )
    else:
        scheme = analyse_bonding(structure)
    apply_overrides(
        scheme,
        polyhedra=_pairs(args.poly) + poly_pairs,
        bonds=_pairs(args.bond),
        no_bonds=_pairs(args.nobond),
    )

    formula = structure.composition.reduced_formula
    lines = [f"{path}  ({formula}, {len(structure)} sites)"]
    if scheme.oxidation_states:
        ox = "  ".join(
            f"{el}{q:+g}" for el, q in sorted(
                scheme.oxidation_states.items(), key=lambda kv: -kv[1],
            )
        )
        lines.append(f"  oxidation states: {ox}")
    lines += [f"  {line}" for line in scheme.describe()]

    # Validate every view before rendering any of them.
    directions = {
        view: _view_direction(structure, view) for view, _fmt in outputs
    }
    if args.dry_run:
        lines += [f"  -> {out} (not written)" for out in outputs.values()]
        return lines

    scene = from_pymatgen(
        structure,
        bond_specs=scheme.bond_specs(),
        polyhedra=scheme.polyhedra(),
    )
    for element, entry in colours.items():
        style = scene.atom_styles.get(element)
        if style is None:
            continue
        if "colour" in entry:
            style.colour = entry["colour"]
        if "radius" in entry:
            style.radius = entry["radius"]

    # Polyhedron and hydrogen completion pull in periodic copies of
    # atoms; drop the leftover copies on the other side of the cell.
    style_kwargs: dict = {"drop_orphan_copies": True}
    if args.no_cell:
        style_kwargs["show_cell"] = False
    if args.no_axes:
        style_kwargs["show_axes"] = False
    style = (
        RenderStyle(**style_kwargs) if args.flat
        else RenderStyle.glossy(**style_kwargs)
    )
    for (view, _fmt), out in outputs.items():
        direction, up = directions[view]
        scene.view.look_along(direction, up=up)
        out.parent.mkdir(parents=True, exist_ok=True)
        scene.render_mpl(
            out, figsize=(args.size, args.size), dpi=args.dpi, style=style,
        )
        label = VIEWS.get(view, f"[{view}]")
        lines.append(f"  -> {out}  ({label})")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``hofmann`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        import pymatgen  # noqa: F401
    except ImportError:
        parser.error(
            "the hofmann command needs pymatgen to read structure files; "
            'install it with: pip install "hofmann[pymatgen]"'
        )

    import matplotlib

    matplotlib.use("Agg")

    parsed = parse_inputs(args.inputs, parser)
    views = list(parsed.views)
    for extra in args.view or []:
        if _VIEW_WORD.match(extra):
            views += [v for v in extra if v not in views]
        else:
            try:
                _parse_direction(extra)
            except ValueError as exc:
                parser.error(str(exc))
            views.append(extra)
    if not views:
        views = list(VIEWS)
    formats = list(parsed.formats)
    formats += [f for f in (args.formats or []) if f not in formats]
    if not formats:
        formats = ["png"]

    paths = parsed.paths or [Path.cwd()]
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = find_structure_files(path)
            if not found:
                print(f"no structure files found in {path}", file=sys.stderr)
            files.extend(found)
        else:
            files.append(path)
    if not files:
        return 1

    colours: dict[str, dict] = {}
    colour_files = (
        [args.colours] if args.colours is not None
        else _default_colour_files(Path.cwd())
    )
    for colour_file in colour_files:
        try:
            colours.update(load_colour_overrides(colour_file))
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read colours from {colour_file}: {exc}")

    used: set[Path] = set()
    failures = 0
    for path in files:
        outputs = {
            (view, fmt): _output_path(path, args.outdir, view, fmt, used)
            for view in views
            for fmt in formats
        }
        try:
            lines = render_file(path, outputs, args, parsed.poly_pairs, colours)
        except Exception as exc:  # report and carry on with other files
            failures += 1
            print(f"{path}: error: {exc}", file=sys.stderr)
            continue
        print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
