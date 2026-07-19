"""Make ``src/`` importable for tests run with a bare ``pytest`` invocation.

``pyproject.toml`` already sets ``pythonpath``, but this keeps tests working
even when someone runs pytest from a checkout without installing the package.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
