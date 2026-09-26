"""ActiveEvaluator: budgeted meta-dataset selection for label-free model evaluation."""

from .distances import pairwise
from .evaluator import MetaEvaluator, Task
from .records import ModelRecord
from .selection import Selection, facility_location, select, similarity

__all__ = ["MetaEvaluator", "ModelRecord", "Selection", "Task", "facility_location", "pairwise", "select", "similarity"]
