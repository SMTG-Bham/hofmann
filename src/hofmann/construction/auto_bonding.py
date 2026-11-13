"""Automatic choice of bonds and polyhedra for ionic structures.

For an ionic compound, drawing every cation-anion bond usually gives
a cluttered picture.  Crystal chemists instead draw the strongly
bonded units (PS4 tetrahedra, ZrO6 octahedra, ...) as polyhedra,
show weaker framework bonds as sticks, and leave the most weakly
bound (typically mobile) cations unbonded.

This module makes that choice automatically:

1. Guess oxidation states from the composition.  If no charge-balanced
   assignment with any anions exists, the structure is treated as
   metallic and every nearest-neighbour bond is drawn.
2. For each cation species, estimate a bond strength
   ``|q_cation * q_anion| / d``, where *d* is the mean cation-anion
   distance within the bond cutoff, taking the strongest anion
   partner.  Like charges never bond.
3. Rank cations from strongest to weakest and assign display tiers
   by the number of cation species *n*:

   ====  ==============================================
   n     strongest  ->  weakest
   ====  ==============================================
   1     polyhedra (bonds if the compound is binary)
   2     polyhedra, hidden
   3     polyhedra, bonds, hidden
   >= 4  polyhedra ..., bonds, hidden
   ====  ==============================================

   With a single anion species this is the familiar rule "ternary:
   strong bonds as polyhedra, weak bonds hidden; quaternary: strong
   as polyhedra, medium as bonds, weak hidden".

   Hydrogen forms covalent bonds rather than acting as a spectator
   cation, so its bonds are always drawn and it is left out of the
   ranking.

The result can be adjusted pair by pair before conversion to
:class:`~hofmann.BondSpec` and :class:`~hofmann.PolyhedronSpec` rules.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from hofmann.construction.defaults import (
    COVALENT_RADII,
    _load_vesta_cutoffs,
)
from hofmann.model import BondSpec, PolyhedronSpec

if TYPE_CHECKING:
    from pymatgen.core import Structure

# Formal charges used when no charge-balanced oxidation-state
# assignment exists (e.g. off-stoichiometric slabs).  Only elements
# listed here can be treated as anions by the fallback.
_FALLBACK_ANION_CHARGES: dict[str, float] = {
    "N": -3.0, "O": -2.0, "S": -2.0, "Se": -2.0, "Te": -2.0,
    "F": -1.0, "Cl": -1.0, "Br": -1.0, "I": -1.0,
}

# Scale applied to the sum of covalent radii for pairs without a
# tabulated bond cutoff.
_COVALENT_CUTOFF_SCALE = 1.25

# Metallic structures: bonds up to this multiple of the shortest
# interatomic distance (the nearest-neighbour shell).
_METALLIC_SHELL_SCALE = 1.1

# Default face opacity of automatically generated polyhedra.
_POLYHEDRON_ALPHA = 0.5


class PairDisplay(StrEnum):
    """How a cation-anion (or any) species pair is drawn.

    Attributes:
        POLYHEDRA: Bonds are drawn and the centre species gets a
            coordination polyhedron.
        BOND: Bonds are drawn as sticks.
        NONE: No bonds are drawn between the pair.
    """

    POLYHEDRA = "polyhedra"
    BOND = "bond"
    NONE = "none"


@dataclass
class PairRule:
    """Display rule for one species pair.

    Attributes:
        species: The pair, as ``(centre, other)`` for polyhedra.
        display: How the pair is drawn.
        max_length: Bond length cutoff in angstroms.
        strength: Estimated ``|q_a q_b| / d`` strength, or ``None``
            when not computed (metallic structures, manual pairs).
        mean_length: Mean bonded distance, or ``None``.
    """

    species: tuple[str, str]
    display: PairDisplay
    max_length: float
    strength: float | None = None
    mean_length: float | None = None


@dataclass
class BondingScheme:
    """Bond and polyhedron choices for one structure.

    Attributes:
        oxidation_states: Oxidation state per element.  Empty for a
            structure treated as metallic.
        rules: Rules keyed by the sorted species pair.
        metallic: Whether the structure was treated as metallic, in
            which case every default bond is drawn.
    """

    oxidation_states: dict[str, float]
    rules: dict[tuple[str, str], PairRule] = field(default_factory=dict)
    metallic: bool = False

    def _cation_first(self, a: str, b: str) -> tuple[str, str]:
        qa = self.oxidation_states.get(a, 0.0)
        qb = self.oxidation_states.get(b, 0.0)
        if qb > 0 and not qa > 0:
            return b, a
        return a, b

    def set_pair(self, a: str, b: str, display: PairDisplay | str) -> None:
        """Override how the pair ``(a, b)`` is drawn.

        For :attr:`PairDisplay.POLYHEDRA` the polyhedron is centred on
        the cation of the pair when oxidation states are known, and
        on *a* otherwise.  Setting a polyhedron pair to bonds turns
        it back into sticks (``--bond Zr O`` on ZrO6 octahedra).

        Args:
            a: First species.
            b: Second species.
            display: New display mode.
        """
        display = PairDisplay(display)
        key = tuple(sorted((a, b)))
        centre, other = (
            self._cation_first(a, b) if self.oxidation_states else (a, b)
        )
        existing = self.rules.get(key)
        if existing is not None:
            existing.display = display
            existing.species = (centre, other)
        else:
            self.rules[key] = PairRule(
                species=(centre, other),
                display=display,
                max_length=_pair_cutoff(a, b),
            )

    def bond_specs(self) -> list[BondSpec]:
        """Return bond rules for every pair drawn as bonds or polyhedra.

        Bonds to hydrogen complete the partner atom's shell across
        periodic boundaries, so groups such as NH3 or OH are drawn
        whole instead of losing hydrogens to the far side of the
        cell.  Render with ``drop_orphan_copies=True`` to drop the
        leftover copies.
        """
        specs = []
        for rule in self.rules.values():
            if rule.display == PairDisplay.NONE:
                continue
            a, b = rule.species
            complete: str | bool = False
            if "H" in (a, b) and a != b:
                complete = b if a == "H" else a
            specs.append(BondSpec(
                species=rule.species,
                max_length=rule.max_length,
                complete=complete,
            ))
        return specs

    def polyhedra(self) -> list[PolyhedronSpec]:
        """Return one polyhedron rule per centre species."""
        centres = sorted({
            rule.species[0] for rule in self.rules.values()
            if rule.display == PairDisplay.POLYHEDRA
        })
        return [
            PolyhedronSpec(centre=c, alpha=_POLYHEDRON_ALPHA)
            for c in centres
        ]

    def describe(self) -> list[str]:
        """Human-readable summary lines, strongest pair first."""
        lines: list[str] = []
        if self.metallic:
            lines.append("metallic: all nearest-neighbour bonds")
        rules = sorted(
            self.rules.values(),
            key=lambda r: -(r.strength if r.strength is not None else np.inf),
        )
        for rule in rules:
            a, b = rule.species
            if rule.strength is not None and rule.mean_length is not None:
                detail = (
                    f"strength {rule.strength:.2f}, "
                    f"d = {rule.mean_length:.2f} A"
                )
            elif rule.mean_length is not None:
                detail = f"d = {rule.mean_length:.2f} A"
            else:
                detail = "manual"
            lines.append(f"{a}-{b}: {rule.display.value} ({detail})")
        return lines


def _pair_cutoff(a: str, b: str) -> float:
    """Bond length cutoff for a species pair."""
    key = tuple(sorted((a, b)))
    cutoffs = _load_vesta_cutoffs()
    if key in cutoffs:
        return float(cutoffs[key])
    return _COVALENT_CUTOFF_SCALE * (
        COVALENT_RADII.get(a, 1.0) + COVALENT_RADII.get(b, 1.0)
    )


def _site_elements(site) -> dict[str, float]:
    """Element symbol -> occupancy for a pymatgen site."""
    occ: dict[str, float] = defaultdict(float)
    for sp, amount in site.species.items():
        occ[sp.symbol] += float(amount)
    return dict(occ)


def guess_oxidation_states(structure: Structure) -> dict[str, float]:
    """Guess one oxidation state per element of *structure*.

    Uses pymatgen's charge-balancing guess on the composition.  When no
    charge-balanced assignment exists (common for off-stoichiometric
    slabs), falls back to typical formal charges for common anions
    and the highest common oxidation state for every other element.

    Returns:
        Element symbol -> oxidation state.  Empty when the structure
        has no anions (a metal or alloy).
    """
    from pymatgen.core import Element

    comp = structure.composition.element_composition
    elements = [el.symbol for el in comp.elements]
    try:
        guesses = comp.oxi_state_guesses(max_sites=-1)
    except Exception:
        guesses = ()
    if guesses:
        states = {el: float(guesses[0].get(el, 0.0)) for el in elements}
        if any(q < 0 for q in states.values()):
            return states
        return {}

    anions = {
        el: q for el, q in _FALLBACK_ANION_CHARGES.items() if el in elements
    }
    if not anions or len(anions) == len(elements):
        return {}
    states: dict[str, float] = dict(anions)
    for el in elements:
        if el in states:
            continue
        common = [q for q in Element(el).common_oxidation_states if q > 0]
        states[el] = float(max(common)) if common else 0.0
    return states


def _display_tiers(n_cations: int, n_elements: int) -> list[PairDisplay]:
    """Display tier for each cation, strongest first."""
    if n_cations <= 0:
        return []
    if n_cations == 1:
        return [
            PairDisplay.POLYHEDRA if n_elements >= 3 else PairDisplay.BOND,
        ]
    if n_cations == 2:
        return [PairDisplay.POLYHEDRA, PairDisplay.NONE]
    return (
        [PairDisplay.POLYHEDRA] * (n_cations - 2)
        + [PairDisplay.BOND, PairDisplay.NONE]
    )


def analyse_bonding(structure: Structure) -> BondingScheme:
    """Choose bonds and polyhedra for *structure* automatically.

    See the module docstring for the rules.

    Args:
        structure: A pymatgen ``Structure`` (ordered or disordered).

    Returns:
        A :class:`BondingScheme`.  Adjust it with
        :meth:`BondingScheme.set_pair`, then pass
        :meth:`BondingScheme.bond_specs` and
        :meth:`BondingScheme.polyhedra` to
        :func:`~hofmann.from_pymatgen`.
    """
    elements = sorted(
        el.symbol for el in structure.composition.element_composition.elements
    )
    ox = guess_oxidation_states(structure)
    if not ox:
        return _metallic_scheme(structure)

    # Hydrogen bonds covalently: always drawn, never ranked.
    cations = [el for el in elements if ox[el] > 0]
    anions = [el for el in elements if ox[el] < 0]
    cutoffs = {
        (c, a): _pair_cutoff(c, a) for c in cations for a in anions
    }
    scheme = BondingScheme(oxidation_states=ox)
    if not cutoffs:
        return scheme

    # Collect cation-anion distances within each pair's cutoff.
    r_max = max(cutoffs.values())
    centres, neighbours, _images, distances = structure.get_neighbor_list(
        r_max,
    )
    site_els = [_site_elements(site) for site in structure]
    sums: dict[tuple[str, str], float] = defaultdict(float)
    weights: dict[tuple[str, str], float] = defaultdict(float)
    for i, j, d in zip(centres, neighbours, distances):
        for el_i, occ_i in site_els[i].items():
            if ox.get(el_i, 0.0) <= 0:
                continue
            for el_j, occ_j in site_els[j].items():
                key = (el_i, el_j)
                if key in cutoffs and d <= cutoffs[key]:
                    w = occ_i * occ_j
                    sums[key] += w * d
                    weights[key] += w

    # Strength of each bonded pair, and of each cation (its strongest).
    pair_info: dict[tuple[str, str], tuple[float, float]] = {}
    for key, w in weights.items():
        if w <= 0:
            continue
        mean_d = sums[key] / w
        c, a = key
        pair_info[key] = (abs(ox[c] * ox[a]) / mean_d, mean_d)
    cation_strength: dict[str, float] = {}
    for (c, _a), (strength, _d) in pair_info.items():
        cation_strength[c] = max(cation_strength.get(c, 0.0), strength)

    ranked = sorted(
        (c for c in cation_strength if c != "H"),
        key=lambda c: -cation_strength[c],
    )
    tiers = dict(zip(ranked, _display_tiers(len(ranked), len(elements))))
    if "H" in cation_strength:
        tiers["H"] = PairDisplay.BOND

    for (c, a), (strength, mean_d) in pair_info.items():
        scheme.rules[tuple(sorted((c, a)))] = PairRule(
            species=(c, a),
            display=tiers[c],
            max_length=cutoffs[(c, a)],
            strength=strength,
            mean_length=mean_d,
        )
    return scheme


def _metallic_scheme(structure: Structure) -> BondingScheme:
    """Bond every pair found within the nearest-neighbour shell."""
    scheme = BondingScheme(oxidation_states={}, metallic=True)
    if len(structure) < 1:
        return scheme
    # Shortest interatomic distance, searching out to a generous radius.
    centres, neighbours, _images, distances = structure.get_neighbor_list(
        6.0,
    )
    if len(distances) == 0:
        return scheme
    cutoff = _METALLIC_SHELL_SCALE * float(np.min(distances))
    site_els = [_site_elements(site) for site in structure]
    pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for i, j, d in zip(centres, neighbours, distances):
        if d > cutoff:
            continue
        for el_i in site_els[i]:
            for el_j in site_els[j]:
                pairs[tuple(sorted((el_i, el_j)))].append(float(d))
    for (a, b), ds in pairs.items():
        scheme.rules[(a, b)] = PairRule(
            species=(a, b),
            display=PairDisplay.BOND,
            max_length=cutoff,
            mean_length=float(np.mean(ds)),
        )
    return scheme


def apply_overrides(
    scheme: BondingScheme,
    *,
    polyhedra: Iterable[tuple[str, str]] = (),
    bonds: Iterable[tuple[str, str]] = (),
    no_bonds: Iterable[tuple[str, str]] = (),
) -> BondingScheme:
    """Apply manual pair overrides to *scheme* in place.

    Overrides are applied in the order bonds, polyhedra, no-bonds, so
    an explicit ``no_bonds`` entry always wins.

    Returns:
        The same *scheme*, for chaining.
    """
    for a, b in bonds:
        scheme.set_pair(a, b, PairDisplay.BOND)
    for a, b in polyhedra:
        scheme.set_pair(a, b, PairDisplay.POLYHEDRA)
    for a, b in no_bonds:
        scheme.set_pair(a, b, PairDisplay.NONE)
    return scheme
