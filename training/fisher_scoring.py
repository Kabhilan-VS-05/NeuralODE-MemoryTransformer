"""
training/fisher_scoring.py
--------------------------
Diagonal Empirical Fisher-Information Memory Scoring.

Computes parameter-sensitivity importance scores for memory candidates:
    S_Fisher(x_i) = sum_{p in Theta} ( dL(theta; x_i, y_i) / d_theta_p )^2 = ||grad_theta L(theta; x_i, y_i)||_2^2

Samples with high gradient energy (Fisher information trace) represent critical anchor points
whose removal would cause maximum loss curvature displacement if overwritten.
"""

import torch
import torch.nn.functional as F

_PROTOTYPE_STATE = {}


def _get_target_params(model, task_id: int = 0):
    """
    Returns the parameter list to score w.r.t.:
    - For shared 100-way head: model.classifier (and model.proj if dinov2)
    - For multi-head: model.classifiers[task_id]
    """
    if getattr(model, 'head', 'multi') == 'shared':
        return list(model.classifier.parameters())
    else:
        return list(model.classifiers[task_id].parameters())


def compute_fisher_scores(model, candidate_x: torch.Tensor, candidate_y: torch.Tensor,
                          task_id: int = 0) -> torch.Tensor:
    """
    Computes per-sample diagonal Fisher Information scores for candidate samples.
    
    Args:
        model: HybridModel instance
        candidate_x: (N, feature_dim) tensor of candidate features
        candidate_y: (N,) tensor of candidate target labels
        task_id: task index (for multi-head routing)
        
    Returns:
        scores: (N,) tensor of Fisher importance scores (higher = more critical to retain)
    """
    model.eval()
    params = _get_target_params(model, task_id=task_id)
    scores = []

    for i in range(candidate_x.shape[0]):
        model.zero_grad(set_to_none=True)
        xi = candidate_x[i:i + 1]
        yi = candidate_y[i:i + 1]
        
        if getattr(model, 'head', 'multi') == 'shared':
            logits = model(xi, write_memory=False)
            loss = F.cross_entropy(logits, yi)
        else:
            yi_local = yi - task_id * model.classes_per_task
            logits = model(xi, write_memory=False, task_id=task_id)
            loss = F.cross_entropy(logits, yi_local)
            
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        flat_grad = torch.cat([g.flatten() for g in grads])
        
        # Diagonal Fisher score: sum of squared gradients = squared L2 norm
        fisher_val = torch.sum(flat_grad ** 2)
        scores.append(fisher_val.detach())

    model.train()
    return torch.stack(scores, dim=0)


def score_memory_candidates(model, candidate_x: torch.Tensor, candidate_y: torch.Tensor,
                             new_task_x: torch.Tensor = None, new_task_y: torch.Tensor = None,
                             candidate_task_id: int = 0,
                             new_task_id: int = 0) -> torch.Tensor:
    """
    Drop-in alternative signature to influence_scoring.score_memory_candidates.
    new_task_x and new_task_y are accepted for interface compatibility but not needed
    since Fisher Information is intrinsic to the current task manifold.
    """
    return compute_fisher_scores(model, candidate_x, candidate_y, task_id=candidate_task_id)


def compute_fisher_proto_scores(model, candidate_x: torch.Tensor, candidate_y: torch.Tensor,
                                task_id: int = 0, w1: float = 0.5, w2: float = 0.5) -> torch.Tensor:
    """
    Computes Prototype-Aware Fisher-Information scores for candidates.
    combines fisher info (sensitivity) with prototype distance (representativeness).
    """
    global _PROTOTYPE_STATE
    
    # 1. Update prototypes with current candidates cumulatively
    for i in range(candidate_x.shape[0]):
        cx = candidate_x[i].detach()
        cy = int(candidate_y[i].item())
        if cy not in _PROTOTYPE_STATE:
            _PROTOTYPE_STATE[cy] = {'sum': torch.zeros_like(cx), 'count': 0}
        _PROTOTYPE_STATE[cy]['sum'] += cx
        _PROTOTYPE_STATE[cy]['count'] += 1
        
    # 2. Get raw fisher scores
    fisher_raw = compute_fisher_scores(model, candidate_x, candidate_y, task_id)
    
    # 3. Get raw distances to prototype
    dist_raw = torch.zeros_like(fisher_raw)
    for i in range(candidate_x.shape[0]):
        cx = candidate_x[i]
        cy = int(candidate_y[i].item())
        proto = _PROTOTYPE_STATE[cy]['sum'] / _PROTOTYPE_STATE[cy]['count']
        dist_raw[i] = torch.norm(cx - proto, p=2)
        
    # 4. Min-max normalization
    def min_max_norm(t):
        t_max, t_min = t.max(), t.min()
        if t_max == t_min:
            return torch.zeros_like(t)
        return (t - t_min) / (t_max - t_min + 1e-8)
        
    fisher_norm = min_max_norm(fisher_raw)
    dist_norm = min_max_norm(dist_raw)
    
    # 5. Combine using custom weights
    combined_scores = w1 * fisher_norm - w2 * dist_norm
    return combined_scores


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.hybrid import HybridModel

    print("--- Testing Fisher Scoring (Shared Head) ---")
    model_s = HybridModel(backbone="dinov2", head="shared")
    cand_x = torch.randn(8, 384)
    cand_y = torch.randint(0, 10, (8,))
    scores_s = compute_fisher_scores(model_s, cand_x, cand_y, task_id=0)
    print("Fisher scores shape:", scores_s.shape)
    print("Fisher scores values:", scores_s)

    print("\n--- Testing Fisher Scoring (Multi Head) ---")
    model_m = HybridModel(backbone="dinov2", head="multi")
    scores_m = compute_fisher_scores(model_m, cand_x, cand_y, task_id=0)
    print("Fisher scores shape:", scores_m.shape)
    print("Fisher scores values:", scores_m)

    print("\n--- Testing Fisher Proto Scoring ---")
    scores_p = compute_fisher_proto_scores(model_s, cand_x, cand_y, task_id=0)
    print("Fisher Proto scores shape:", scores_p.shape)
    print("Fisher Proto scores values:", scores_p)
    print("Prototype state keys:", _PROTOTYPE_STATE.keys())
