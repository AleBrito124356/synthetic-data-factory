#!/usr/bin/env python3
"""Checkout convenience wrapper: ``python cli.py ...`` == ``sdf ...``.

The real command line lives in :mod:`factory.cli` so it ships inside the
package (the ``sdf`` console script and ``python -m factory`` use it too).
This shim only makes ``src/`` importable when the package is not installed.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from factory.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
