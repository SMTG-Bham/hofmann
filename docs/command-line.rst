Command line
============

Installing hofmann with pymatgen also installs a ``hofmann`` command
that turns structure files into images without writing any Python:

.. code-block:: bash

   pip install "hofmann[pymatgen]"

   hofmann                    # every structure file in this directory
   hofmann Li3PS4.cif         # one file
   hofmann runs/              # every structure file in runs/
   hofmann x P S Li3PS4.cif   # P-S polyhedra, viewed along [100] only
   hofmann x P S svg Li3PS4.cif   # the same, as SVG

CIF files, VASP ``POSCAR`` / ``CONTCAR`` files (with or without a
suffix), and files ending in ``.vasp`` or ``.poscar`` are recognised.
Images use the :meth:`~hofmann.RenderStyle.glossy` preset; pass
``--flat`` for the classic flat style.  ``python -m hofmann`` is
equivalent to ``hofmann``.

Arguments and views
-------------------

The positional arguments can come in any order; each is recognised by
its form:

* a word made only of the lower-case letters ``w``, ``x``, ``y``,
  ``z`` chooses the views:

  .. list-table::
     :widths: 15 85

     * - ``x``
       - along [100] (the *a* axis), *c* up
     * - ``y``
       - along [010] (the *b* axis), *c* up
     * - ``z``
       - along [001] (the *c* axis), *b* up
     * - ``w``
       - an oblique view showing all three axes

  Letters combine, so ``xz`` gives two images.  Without a view word,
  all four are drawn.
* an element symbol (capitalised, such as ``P`` or ``Li``).  Element
  symbols are read in pairs, and each pair is drawn as polyhedra
  (see ``--poly`` below);
* ``png``, ``svg`` or ``pdf``: the image format.  Several can be
  given (``svg pdf``) to write each view in every format.  The default
  is ``png``;
* an existing file or directory.

A natural order is ``hofmann VIEWS ELEMENTS FORMAT FILES``, for
example ``hofmann xw P S pdf Li3PS4.cif``, but any order works.

Each view is written next to its input as ``name_x.png``,
``name_y.png``, ``name_z.png``, and ``name_w.png``.  For any other
direction, use ``--view u,v,w`` (for example ``--view 1,1,0`` writes
``name_1-1-0.png``).

For each file, hofmann prints the bonding it chose::

   LLZO.cif  (Li7La3Zr2O12, 192 sites)
     oxidation states: Zr+4  La+3  Li+1  O-2
     Zr-O: polyhedra (strength 3.75, d = 2.13 A)
     La-O: bond (strength 2.34, d = 2.57 A)
     Li-O: none (strength 0.94, d = 2.12 A)
     -> LLZO.png

Automatic bonds and polyhedra
-----------------------------

For ionic compounds, hofmann decides which bonds to draw:

1. Oxidation states are guessed from the composition.  Atoms with
   charges of the same sign never bond.
2. Each cation species gets a rough bond strength
   ``|q_cation * q_anion| / d``, with *d* the mean cation-anion
   distance, using its strongest anion partner.
3. Cations are ranked from strongest to weakest and drawn according
   to the number of cation species:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Cation species
        - Strongest to weakest
      * - 1
        - polyhedra (bonds for a binary compound)
      * - 2 (e.g. Li3PS4)
        - polyhedra, no bonds
      * - 3 (e.g. Li7La3Zr2O12)
        - polyhedra, bonds, no bonds
      * - 4 or more
        - polyhedra for all but the two weakest, then bonds, then
          no bonds

So for Li3PS4 the PS4 tetrahedra are drawn as polyhedra and Li is left
unbonded; for LLZO the ZrO6 octahedra are polyhedra, La-O bonds are
shown as sticks, and Li is unbonded.

Hydrogen always keeps its bonds, so molecular cations and ligands stay
intact.  Structures without anions (metals and alloys) show every bond
in the nearest-neighbour shell.

Polyhedra (and hydrogen-containing groups) are completed across
periodic boundaries: a vertex that lies in a neighbouring cell is
drawn there, next to its centre.  The original copy of that atom on
the other side of the cell would otherwise be left floating without
bonds, so such copies are removed (``drop_orphan_copies`` on
:class:`~hofmann.RenderStyle`).  Atoms that have no bonds anywhere,
such as unbonded Li or the corner atoms of the cell, are kept.

The estimate is deliberately simple.  Use ``--dry-run`` (``-n``) to
see the choice without rendering, and override it where needed.

Choosing bonds by hand
----------------------

Each override takes a pair of elements and can be repeated:

``--poly A B``
    Draw A-B bonds as polyhedra, centred on the cation of the pair.
``--bond A B``
    Draw A-B bonds as sticks (also turns automatic polyhedra into
    sticks).
``--nobond A B``
    Do not draw A-B bonds.

Element pairs among the positional arguments are shorthand for
``--poly``:

.. code-block:: bash

   hofmann Li3PS4.cif P S                    # same as --poly P S
   hofmann x P S Li3PS4.cif                  # the same, [100] view only
   hofmann Li3PS4.cif --poly P S --nobond Li S
   hofmann LLZO.cif --bond Zr O              # octahedra as sticks
   hofmann LLZO.cif --no-auto --poly Zr O    # only the bonds you ask for

``--no-auto`` skips the automatic analysis and starts from every
default bond instead.

Element colours
---------------

Elements use hofmann's default palette
(:data:`~hofmann.ELEMENT_COLOURS`).  To change colours, put a YAML (or
JSON) file mapping elements to colours in the working directory as
``hofmann_colours.yaml``, in ``~/.config/hofmann/colours.yaml``, or
pass one with ``--colours``.  Any colour format accepted by
:func:`~hofmann.normalise_colour` works, and a radius can be set too:

.. code-block:: yaml

   Li: "#7bc86c"
   P: [0.75, 0.55, 0.85]
   S: gold
   La: {colour: seagreen, radius: 1.4}

The file in the working directory takes precedence over the one in
``~/.config``.

Other options
-------------

``-o DIR``, ``--outdir DIR``
    Write images to *DIR* instead of next to each input.
``-f {png,svg,pdf}``, ``--format``
    Image format, same as the positional format words (repeatable;
    default ``png``).
``--dpi N``, ``--size INCHES``
    Resolution (default 300) and figure size (default 6).
``--view U,V,W``
    Add a view along the lattice direction ``[uvw]`` (repeatable).
    Also accepts ``wxyz`` letters.
``--supercell NA NB NC``
    Build a supercell first.
``--flat``
    Flat shading instead of glossy.
``--no-cell``, ``--no-axes``
    Hide the unit cell or the axes widget.
``-n``, ``--dry-run``
    Print the chosen bonding without rendering.

Run ``hofmann --help`` for the full list.
