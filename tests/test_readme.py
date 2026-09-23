"""Every complete schema shown in the README must generate and validate,
and the library snippet's imports must exist."""
import os
import re

import pytest
import yaml

from factory import generate, validate_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _yaml_blocks():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        text = fh.read()
    blocks = [yaml.safe_load(b) for b in re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)]
    return [b for b in blocks if isinstance(b, dict) and "tables" in b]


def test_readme_has_schema_examples():
    assert _yaml_blocks()


@pytest.mark.parametrize("index", range(len(_yaml_blocks())))
def test_readme_schema_generates_and_validates(index):
    dataset = generate(_yaml_blocks()[index])
    report = validate_dataset(dataset)
    assert report.ok, report.render()


def test_readme_library_imports_exist():
    from factory import (  # noqa: F401
        RecordingClient, ReplayClient, export_dataset, infer_schema, load_raw,
        validate_data, validate_records,
    )
