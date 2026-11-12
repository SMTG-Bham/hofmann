"""Tests for automatic bond / polyhedron selection."""

from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from hofmann.construction.auto_bonding import (  # noqa: E402
    BondingScheme,
    PairDisplay,
    _display_tiers,
    analyse_bonding,
    apply_overrides,
    guess_oxidation_states,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _rule(scheme, a, b):
    return scheme.rules[tuple(sorted((a, b)))]


@pytest.fixture(scope="module")
def srtio3():
    return Structure.from_file(str(EXAMPLES / "srtio3.cif"))


@pytest.fixture(scope="module")
def llzo():
    return Structure.from_file(str(EXAMPLES / "Li7La3Zr2O12.cif"))


def _nacl():
    return Structure(
        Lattice.cubic(5.64), ["Na"] * 4 + ["Cl"] * 4,
        [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
         [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]],
    )


def _copper():
    return Structure(
        Lattice.cubic(3.61), ["Cu"] * 4,
        [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
    )


def _ammonium_chloride():
    h = 0.154
    return Structure(
        Lattice.cubic(3.87), ["N", "Cl", "H", "H", "H", "H"],
        [[0, 0, 0], [.5, .5, .5],
         [h, h, h], [-h, -h, h], [-h, h, -h], [h, -h, -h]],
    )


class TestDisplayTiers:
    def test_binary_is_bonds(self):
        assert _display_tiers(1, 2) == [PairDisplay.BOND]

    def test_single_cation_multiple_anions_is_polyhedra(self):
        assert _display_tiers(1, 3) == [PairDisplay.POLYHEDRA]

    def test_ternary(self):
        assert _display_tiers(2, 3) == [PairDisplay.POLYHEDRA, PairDisplay.NONE]

    def test_quaternary(self):
        assert _display_tiers(3, 4) == [
            PairDisplay.POLYHEDRA, PairDisplay.BOND, PairDisplay.NONE,
        ]

    def test_five_or_more(self):
        assert _display_tiers(4, 5) == [
            PairDisplay.POLYHEDRA, PairDisplay.POLYHEDRA,
            PairDisplay.BOND, PairDisplay.NONE,
        ]


class TestOxidationStates:
    def test_perovskite(self, srtio3):
        assert guess_oxidation_states(srtio3) == {
            "Sr": 2.0, "Ti": 4.0, "O": -2.0,
        }

    def test_metal_has_none(self):
        assert guess_oxidation_states(_copper()) == {}

    def test_fallback_for_unbalanced_composition(self):
        # NaCl2 cannot be charge-balanced with Cl at -1 and Na at +1;
        # whichever route is taken, Cl must come out as the anion.
        s = Structure(
            Lattice.cubic(4.0), ["Na", "Cl", "Cl"],
            [[0, 0, 0], [.5, .5, .5], [.5, 0, 0]],
        )
        states = guess_oxidation_states(s)
        assert states["Cl"] < 0
        assert states["Na"] > 0


class TestAnalyseBonding:
    def test_ternary_perovskite(self, srtio3):
        scheme = analyse_bonding(srtio3)
        assert _rule(scheme, "Ti", "O").display == PairDisplay.POLYHEDRA
        assert _rule(scheme, "Ti", "O").species == ("Ti", "O")
        assert _rule(scheme, "Sr", "O").display == PairDisplay.NONE
        assert [p.centre for p in scheme.polyhedra()] == ["Ti"]

    def test_quaternary_llzo(self, llzo):
        scheme = analyse_bonding(llzo)
        assert _rule(scheme, "Zr", "O").display == PairDisplay.POLYHEDRA
        assert _rule(scheme, "La", "O").display == PairDisplay.BOND
        assert _rule(scheme, "Li", "O").display == PairDisplay.NONE
        strengths = {
            c: _rule(scheme, c, "O").strength for c in ("Zr", "La", "Li")
        }
        assert strengths["Zr"] > strengths["La"] > strengths["Li"]

    def test_like_charges_never_bond(self, llzo):
        scheme = analyse_bonding(llzo)
        assert ("O", "O") not in scheme.rules
        assert ("La", "Zr") not in scheme.rules
        species = {spec.species for spec in scheme.bond_specs()}
        assert all("O" in pair for pair in species)

    def test_binary_ionic_bonds(self):
        scheme = analyse_bonding(_nacl())
        assert _rule(scheme, "Na", "Cl").display == PairDisplay.BOND
        assert scheme.polyhedra() == []

    def test_metal_bonds_nearest_neighbours(self):
        scheme = analyse_bonding(_copper())
        assert scheme.metallic
        rule = _rule(scheme, "Cu", "Cu")
        assert rule.display == PairDisplay.BOND
        # Nearest neighbour 2.55 A bonded, second shell 3.61 A not.
        assert 2.55 < rule.max_length < 3.61

    def test_hydrogen_bonds_always_drawn(self):
        scheme = analyse_bonding(_ammonium_chloride())
        assert _rule(scheme, "H", "N").display == PairDisplay.BOND

    def test_hydrogen_bonds_complete_partner_shell(self):
        specs = analyse_bonding(_ammonium_chloride()).bond_specs()
        (spec,) = [s for s in specs if "H" in s.species]
        assert spec.complete == "N"

    def test_other_bonds_not_completed(self, llzo):
        assert all(
            s.complete is False for s in analyse_bonding(llzo).bond_specs()
        )

    def test_bond_specs_exclude_hidden_pairs(self, srtio3):
        specs = analyse_bonding(srtio3).bond_specs()
        assert [s.species for s in specs] == [("O", "Ti")]

    def test_describe_lists_strongest_first(self, llzo):
        lines = analyse_bonding(llzo).describe()
        assert lines[0].startswith("Zr-O: polyhedra")
        assert lines[-1].startswith("Li-O: none")


class TestOverrides:
    def test_poly_centres_on_cation_regardless_of_order(self, llzo):
        scheme = apply_overrides(analyse_bonding(llzo), polyhedra=[("O", "La")])
        assert _rule(scheme, "La", "O").species == ("La", "O")
        assert {p.centre for p in scheme.polyhedra()} == {"Zr", "La"}

    def test_bond_turns_polyhedra_into_sticks(self, llzo):
        scheme = apply_overrides(analyse_bonding(llzo), bonds=[("Zr", "O")])
        assert _rule(scheme, "Zr", "O").display == PairDisplay.BOND
        assert scheme.polyhedra() == []

    def test_nobond_wins(self, llzo):
        scheme = apply_overrides(
            analyse_bonding(llzo),
            polyhedra=[("Li", "O")], no_bonds=[("Li", "O")],
        )
        assert _rule(scheme, "Li", "O").display == PairDisplay.NONE

    def test_new_pair_added(self, srtio3):
        scheme = apply_overrides(analyse_bonding(srtio3), bonds=[("Sr", "Ti")])
        rule = _rule(scheme, "Sr", "Ti")
        assert rule.display == PairDisplay.BOND
        assert rule.max_length > 0
        assert "manual" in "\n".join(scheme.describe())

    def test_metallic_poly_uses_given_order(self):
        scheme = BondingScheme(oxidation_states={}, metallic=True)
        scheme.set_pair("Fe", "Al", "polyhedra")
        assert _rule(scheme, "Fe", "Al").species == ("Fe", "Al")
