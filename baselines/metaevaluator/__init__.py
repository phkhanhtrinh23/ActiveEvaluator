"""Full-budget MetaEvaluator: acquire the entire model x sample-set matrix."""
def select(X, pair_model, pair_sample, target_mask, budget, *, rng, **kw):
    return list(range(len(X)))

__all__ = ["select"]
