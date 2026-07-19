"""Weighted categoricals must match requested proportions closely."""
from collections import Counter

from factory import generate, validate_dataset
from factory.validate import distribution_report


def _weighted_schema(rows=4000):
    return {
        "seed": 3,
        "tables": [
            {
                "name": "t",
                "rows": rows,
                "fields": [
                    {"name": "id", "type": "id"},
                    {
                        "name": "plan",
                        "type": "category",
                        "categories": {"free": 0.6, "pro": 0.3, "enterprise": 0.1},
                    },
                ],
            }
        ],
    }


def test_category_proportions_match_weights():
    dataset = generate(_weighted_schema())
    counts = Counter(r["plan"] for r in dataset["t"])
    total = sum(counts.values())
    expected = {"free": 0.6, "pro": 0.3, "enterprise": 0.1}
    for label, exp in expected.items():
        observed = counts[label] / total
        assert abs(observed - exp) < 0.01, f"{label}: {observed:.3f} vs {exp:.3f}"


def test_all_categories_present():
    dataset = generate(_weighted_schema(rows=100))
    labels = {r["plan"] for r in dataset["t"]}
    assert labels == {"free", "pro", "enterprise"}


def test_validation_distribution_check_passes():
    dataset = generate(_weighted_schema())
    report = validate_dataset(dataset, tolerance=0.02)
    dist_checks = [c for c in report.checks if c.name == "distribution"]
    assert dist_checks and all(c.passed for c in dist_checks), report.render()


def test_distribution_report_shape():
    dataset = generate(_weighted_schema(rows=1000))
    report = distribution_report(dataset)
    plan = report["tables"]["t"]["fields"]["plan"]
    assert plan["type"] == "category"
    assert set(plan["categories"].keys()) == {"free", "pro", "enterprise"}
    # observed and expected are both present per category
    for entry in plan["categories"].values():
        assert "observed" in entry and "expected" in entry


def test_correlation_is_positive():
    schema = {
        "seed": 8,
        "tables": [
            {
                "name": "t",
                "rows": 2000,
                "fields": [
                    {"name": "id", "type": "id"},
                    {"name": "driver", "type": "int", "min": 0, "max": 100},
                    {
                        "name": "target",
                        "type": "float",
                        "min": 0,
                        "max": 1000,
                        "correlate": {"field": "driver", "strength": 0.8},
                    },
                ],
            }
        ],
    }
    rows = generate(schema)["t"]
    xs = [r["driver"] for r in rows]
    ys = [r["target"] for r in rows]
    corr = _pearson(xs, ys)
    assert corr > 0.5, f"expected strong positive correlation, got {corr:.3f}"


def _pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)
