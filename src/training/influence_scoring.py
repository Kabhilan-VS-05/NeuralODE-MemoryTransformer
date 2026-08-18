"""
training/influence_scoring.py
------------------------------
Influence-function memory importance scoring, literal reading of Eq. 15:

    I(x_i) = ||grad_theta L(theta; x_i)||_2
             * E_{x_j ~ D_new}[ cos( grad_theta L(theta;x_i), grad_theta L(theta;x_j) ) ]

Interpretation: samples with HIGH gradient magnitude and NEGATIVE
correlation with new-task gradients are the ones whose removal would
maximally degrade old-task performance -- these should score highest
and be prioritized for retention.

This module belongs to Phase 0 (the base paper's own scoring mechanism).
It is what your later Fisher-Information module (Phase 3 of the extended
roadmap) replaces -- keep this fully working so the comparison is fair.

NOTE: computing a full per-sample gradient w.r.t. all parameters is
expensive. For tractability we score w.r.t. the classifier head's
parameters only (a common, defensible approximation for influence-style
scores at this scale) -- documented here as an assumption, matching the
"if the paper is ambiguous, use the simplest literal reading" rule.
"""

import torch
import torch.nn.functional as F


def _per_sample_grad(model, x: torch.Tensor, y: torch.Tensor,
                     task_id: int = 0) -> torch.Tensor:
    """
    Returns a (batch, num_params) tensor of per-sample gradients w.r.t.
    the classifier head, computed one sample at a time.
    """
    grads = []
    if getattr(model, 'head', 'multi') == 'shared':
        params = list(model.classifier.parameters())
    else:
        params = list(model.classifiers[task_id].parameters())

    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        xi = x[i:i + 1]
        yi = y[i:i + 1]

        if getattr(model, 'head', 'multi') == 'shared':
            logits = model(xi, write_memory=False)
            loss = F.cross_entropy(logits, yi)
        else:
            yi_local = yi - task_id * getattr(model, 'classes_per_task', 10)
            logits = model(xi, write_memory=False, task_id=task_id)
            loss = F.cross_entropy(logits, yi_local)

        g = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        flat = torch.cat([gi.flatten() for gi in g])
        grads.append(flat.detach())
    return torch.stack(grads, dim=0)  # (batch, num_params)


def score_memory_candidates(model, candidate_x: torch.Tensor, candidate_y: torch.Tensor,
                             new_task_x: torch.Tensor = None, new_task_y: torch.Tensor = None,
                             candidate_task_id: int = 0,
                             new_task_id: int = 0) -> torch.Tensor:
    """
    candidate_x/y : samples being considered for retention (from the old-task pool)
    new_task_x/y  : a batch sampled from the incoming new task (D_new in Eq. 15)

    Returns: (num_candidates,) importance scores, Eq. 15.
    Higher score = more important to keep.
    """
    model.eval()
    g_candidates = _per_sample_grad(model, candidate_x, candidate_y, task_id=candidate_task_id)  # (Nc, P)
    
    if new_task_x is not None and new_task_y is not None:
        g_new = _per_sample_grad(model, new_task_x, new_task_y, task_id=new_task_id)            # (Nn, P)
    else:
        # If no future task provided yet, use candidate self-comparison
        g_new = g_candidates

    model.train()

    grad_norms = g_candidates.norm(dim=1)  # (Nc,) -- ||grad L(theta;x_i)||_2

    # cosine similarity between each candidate and every comparison sample
    g_c_norm = F.normalize(g_candidates, dim=1)
    g_n_norm = F.normalize(g_new, dim=1)
    cos_sim = g_c_norm @ g_n_norm.t()          # (Nc, Nn)
    mean_cos = cos_sim.mean(dim=1)              # (Nc,)

    scores = grad_norms * mean_cos              # Eq. 15
    return scores


def select_top_k(candidate_indices, scores: torch.Tensor, k: int):
    """Return the indices of the top-k highest-scoring (most important) samples."""
    k = min(k, scores.numel())
    top_idx = torch.topk(scores, k).indices
    return [candidate_indices[i] for i in top_idx.tolist()]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.hybrid import HybridModel

    print("--- 1. Testing Influence Scoring (DINOv2 Shared Head) ---")
    model_s = HybridModel(backbone="dinov2", head="shared")
    cand_x, cand_y = torch.randn(6, 384), torch.randint(0, 10, (6,))
    new_x, new_y = torch.randn(4, 384), torch.randint(0, 10, (4,))
    scores_s = score_memory_candidates(model_s, cand_x, cand_y, new_x, new_y,
                                       candidate_task_id=0, new_task_id=1)
    print("Influence scores (Shared Head):", scores_s)

    print("\n--- 2. Testing Influence Scoring (DINOv2 Multi Head) ---")
    model_m = HybridModel(backbone="dinov2", head="multi")
    new_y_task1 = torch.randint(10, 20, (4,))
    scores_m = score_memory_candidates(model_m, cand_x, cand_y, new_x, new_y_task1,
                                       candidate_task_id=0, new_task_id=1)
    print("Influence scores (Multi Head):", scores_m)
