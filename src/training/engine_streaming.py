"""
training/engine_streaming.py
-----------------------------
Streaming Continual Learning engine.
Unlike baseline, this engine does not process data in discrete tasks.
Instead, it consumes a continuous stream of batches. Memory is updated periodically
by flushing a "micro-buffer" into the main ReplayBuffer.
"""

import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.hybrid import HybridModel
from src.datasets.data import NUM_TASKS
from src.training.replay_buffer import ReplayBuffer
from src.training.fisher_scoring import compute_fisher_scores, compute_fisher_proto_scores
from src.training.influence_scoring import score_memory_candidates as score_influence
from src.training.engine_baseline import evaluate_task

def flush_micro_buffer(model, micro_buffer_x, micro_buffer_y, replay_buffer, scoring, fisher_w1, fisher_w2, device):
    """
    Scores the accumulated candidates in the micro_buffer and adds them to the main replay_buffer.
    """
    if not micro_buffer_x or replay_buffer is None or scoring == "none":
        return

    cand_x = torch.cat(micro_buffer_x, dim=0).to(device)
    cand_y = torch.cat(micro_buffer_y, dim=0).to(device)

    # For scoring methods that need a task_id, we just pass None or 0 since we're streaming.
    # Our scoring functions should be robust enough, but let's pass a dummy task_id=0 if needed.
    if scoring == "random":
        scores = torch.rand(cand_x.shape[0], device=device)
    elif scoring == "fisher":
        scores = compute_fisher_scores(model, cand_x, cand_y, task_id=0)
    elif scoring == "fisher_proto":
        scores = compute_fisher_proto_scores(model, cand_x, cand_y, task_id=0, w1=fisher_w1, w2=fisher_w2)
    elif scoring == "influence":
        scores = score_influence(model, cand_x, cand_y, candidate_task_id=0)
    else:
        raise ValueError(f"Unknown scoring method '{scoring}'.")

    replay_buffer.add_candidates(cand_x, cand_y, scores)


def run_streaming_sequence(streaming_loader,
                           val_loaders,
                           total_batches: int,
                           lr: float = 5e-4,
                           backbone: str = "dinov2",
                           head: str = "shared",
                           scoring: str = "none",
                           micro_buffer_size: int = 500,
                           replay_batch_size: int = 50,
                           fisher_w1: float = 0.5,
                           fisher_w2: float = 0.5,
                           device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Runs continuous streaming training over the provided generator.
    """
    print(f"=== Streaming Sequence (backbone={backbone}, head={head}, scoring={scoring}) ===")
    
    model = HybridModel(backbone=backbone, head=head).to(device)
    replay_buffer = ReplayBuffer(capacity=500, device=device) if scoring != "none" else None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # We will track accuracy globally over time.
    # We'll evaluate every `eval_every` steps (e.g. 10 times total during the stream)
    eval_every = max(1, total_batches // 10)
    acc_history = []
    
    model.train()
    
    micro_buffer_x = []
    micro_buffer_y = []
    micro_buffer_count = 0
    
    pbar = tqdm(streaming_loader, total=total_batches, desc="Streaming")
    
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        # We assume global labels (0-99) for streaming since tasks are unknown
        y_target = y
        
        # 1. Accumulate into micro-buffer
        if scoring != "none":
            micro_buffer_x.append(x.detach())
            micro_buffer_y.append(y.detach())
            micro_buffer_count += x.shape[0]
            
            # If micro-buffer is full, flush it!
            if micro_buffer_count >= micro_buffer_size:
                flush_micro_buffer(model, micro_buffer_x, micro_buffer_y, replay_buffer, scoring, fisher_w1, fisher_w2, device)
                micro_buffer_x.clear()
                micro_buffer_y.clear()
                micro_buffer_count = 0
                
        # 2. Replay sample mixing
        if replay_buffer is not None and len(replay_buffer) > 0:
            rx, ry = replay_buffer.sample(n=replay_batch_size, device=device)
            if rx is not None and ry is not None:
                x_train = torch.cat([x, rx], dim=0)
                y_train = torch.cat([y_target, ry], dim=0)
            else:
                x_train, y_train = x, y_target
        else:
            x_train, y_train = x, y_target
            
        # 3. Forward & Backward
        optimizer.zero_grad()
        logits = model(x_train, write_memory=True) # head="shared" ignores task_id
        
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()
        model.commit_memory_write()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", buf=f"{len(replay_buffer) if replay_buffer else 0}/500")
        
        # 4. Periodic Evaluation
        if (step + 1) % eval_every == 0 or (step + 1) == total_batches:
            print(f"\n[Step {step+1}/{total_batches}] Running periodic evaluation...")
            # We evaluate on all 10 tasks to see how global accuracy evolves
            eval_accs = []
            for i, v_loader in enumerate(val_loaders):
                acc = evaluate_task(model, v_loader, device, task_id=i)
                eval_accs.append(acc)
            avg_acc = sum(eval_accs) / len(eval_accs)
            acc_history.append({
                "step": step + 1,
                "per_task_acc": eval_accs,
                "avg_acc": avg_acc
            })
            print(f"  --> Global Avg Accuracy: {avg_acc*100:.2f}%")
            model.train() # resume training mode
            
    print("Streaming complete!")
    return acc_history, model
