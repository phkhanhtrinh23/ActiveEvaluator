"""Baseline supervision-acquisition strategies and label-free estimators.

Each baseline lives in its own sub-package and exposes a ``select`` (acquisition)
or ``estimate`` (label-free) callable. The registries below are what
``experiments/run_acquisition_benchmark.py`` iterates over.
"""

from ._core import ACQUISITION_REGISTRY, estimate_atc, estimate_doc

ESTIMATOR_REGISTRY = {"atc": estimate_atc, "doc": estimate_doc}

__all__ = ["ACQUISITION_REGISTRY", "ESTIMATOR_REGISTRY", "estimate_atc", "estimate_doc"]
