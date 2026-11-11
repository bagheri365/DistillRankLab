"""Smoke tests for the initial DistillRankLab packaging scaffold."""

from importlib.metadata import version


def test_editable_distribution_is_installed():
    """The project can be installed with `pip install -e '.[dev]'`."""
    assert version("distillranklab") == "0.1.0"
