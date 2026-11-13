"""Allow ``python -m hofmann`` as an alias for the ``hofmann`` command."""

import sys

from hofmann.cli import main

sys.exit(main())
