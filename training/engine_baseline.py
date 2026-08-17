"""
training/engine_baseline.py
-----------------------------
Standard OFFLINE task-based training loop (predefined task boundaries,
NOT streaming -- streaming is Phase 3+ of the extended roadmap).

After each task, evaluates on ALL tasks seen so far (needed to build the
acc[i, j] matrix that the forgetting metric, Eq. 20, requires:
acc[i, j] = accuracy on task i's val set after training through task j).

This is where the two Phase-0 diagnostics we agreed on live:
  - run_single_task_sanity_check(): trains ONLY task 0, in isolation,
    for a realistic number of epochs, to confirm the architecture can
    learn at all before testing forgetting behavior.
  - run_full_sequence(): the real Phase 0 experiment, run only AFTER
    the sanity check passes.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.hybrid import HybridModel
from data import NUM_TASKS, CLASSES_PER_TASK
from training.replay_buffer import ReplayBuffer
from training.fisher_scoring import compute_fisher_scores, compute_fisher_proto_scores
from training.influence_scoring import score_memory_candidates as score_influence


def get_task_loaders(backbone: str = "dinov2", batch_size: int = 64):
    """
    Dynamically loads the appropriate dataset loader based on the selected backbone:
    - 'dinov2': loads pre-extracted 384-dim features from cached_features/ (via data_cached.py)
    - 'cnn': loads raw 32x32 images for the from-scratch CNN (via data.py)
    """
    if backbone == "dinov2":
        from data_cached import build_task_loaders as build_cached_loaders
        return build_cached_loaders(batch_size=batch_size)
    elif backbone == "cnn":
        from data import build_task_loaders as build_raw_loaders
        return build_raw_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose 'dinov2' or 'cnn'.")


def evaluate_task(model, loader, device, task_id: int) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if getattr(model, 'head', 'multi') == 'shared':
                y_target = y  # global 0..99 (Class-incremental)
                logits = model(x, write_memory=False)
            else:
                y_target = y - task_id * CLASSES_PER_TASK  # local 0..9 (Task-incremental)
                logits = model(x, write_memory=False, task_id=task_id)
            pred = logits.argmax(dim=1)
            correct += (pred == y_target).sum().item()
            total += y.numel()
    model.train()
    return correct / max(total, 1)


def train_one_task(model, loader, optimizer, device, epochs: int, task_id: int,
                   replay_buffer: ReplayBuffer = None, replay_batch_size: int = 50):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"  epoch {epoch + 1}/{epochs}", leave=False)
        running_loss, n_batches = 0.0, 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            if getattr(model, 'head', 'multi') == 'shared':
                y_target = y  # global 0..99
            else:
                y_target = y - task_id * CLASSES_PER_TASK  # local 0..9

            # Replay sample mixing
            if replay_buffer is not None and len(replay_buffer) > 0:
                rx, ry = replay_buffer.sample(n=replay_batch_size, device=device)
                if rx is not None and ry is not None:
                    # Concatenate current task batch with replay batch
                    x_train = torch.cat([x, rx], dim=0)
                    y_train = torch.cat([y_target, ry], dim=0)
                else:
                    x_train, y_train = x, y_target
            else:
                x_train, y_train = x, y_target

            optimizer.zero_grad()
            if getattr(model, 'head', 'multi') == 'shared':
                logits = model(x_train, write_memory=True)
            else:
                logits = model(x_train, write_memory=True, task_id=task_id)

            # Equal weighting across all samples in the combined batch
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()
            model.commit_memory_write()  # write to memory only AFTER the optimizer step
            
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())
        avg_loss = running_loss / max(n_batches, 1)
        print(f"    epoch {epoch + 1}/{epochs}: avg training loss = {avg_loss:.4f}")


def populate_buffer_from_task(model, loader, replay_buffer: ReplayBuffer, scoring: str,
                              task_id: int, device: str, max_candidates: int = 500,
                              fisher_w1: float = 0.5, fisher_w2: float = 0.5):
    """
    Collects candidate samples from completed task, scores them, and adds to buffer.
    """
    if scoring == "none" or replay_buffer is None:
        return

    # Collect up to max_candidates from loader
    all_x, all_y = [], []
    collected = 0
    for x, y in loader:
        all_x.append(x)
        all_y.append(y)
        collected += x.shape[0]
        if collected >= max_candidates:
            break

    cand_x = torch.cat(all_x, dim=0)[:max_candidates].to(device)
    cand_y = torch.cat(all_y, dim=0)[:max_candidates].to(device)

    print(f"  Scoring {cand_x.shape[0]} candidates from Task {task_id} using scoring='{scoring}'...")
    if scoring == "random":
        scores = torch.rand(cand_x.shape[0], device=device)
    elif scoring == "fisher":
        scores = compute_fisher_scores(model, cand_x, cand_y, task_id=task_id)
    elif scoring == "fisher_proto":
        scores = compute_fisher_proto_scores(model, cand_x, cand_y, task_id=task_id, w1=fisher_w1, w2=fisher_w2)
    elif scoring == "influence":
        scores = score_influence(model, cand_x, cand_y, candidate_task_id=task_id)
    else:
        raise ValueError(f"Unknown scoring method '{scoring}'.")

    replay_buffer.add_candidates(cand_x, cand_y, scores)
    print(f"  Buffer size now: {len(replay_buffer)}/500 | Class distribution: {replay_buffer.get_class_counts()}")


def run_single_task_sanity_check(epochs: int = 30, lr: float = 5e-4,
                                  backbone: str = "dinov2",
                                  head: str = "multi",
                                  scoring: str = "none",
                                  batch_size: int = 64,
                                  device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    DIAGNOSTIC 1: train ONLY on task 0.
    """
    print(f"=== Single-task sanity check (task 0 only, {epochs} epochs, backbone={backbone}, head={head}, scoring={scoring}) ===")
    train_loaders, val_loaders = get_task_loaders(backbone=backbone, batch_size=batch_size)
    model = HybridModel(backbone=backbone, head=head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_one_task(model, train_loaders[0], optimizer, device, epochs=1, task_id=0)
        scheduler.step()
        acc = evaluate_task(model, val_loaders[0], device, task_id=0)
        print(f"  epoch {epoch + 1}/{epochs}: task-0 val accuracy = {acc * 100:.2f}%")

    final_acc = evaluate_task(model, val_loaders[0], device, task_id=0)
    print(f"Final task-0 accuracy after {epochs} epochs: {final_acc * 100:.2f}%")
    print("PASS: model learns well above chance (>30%)." if final_acc > 0.30
          else "FAIL: model near chance level -- debug architecture/training "
               "regime BEFORE running the full task sequence.")
    return final_acc


def run_full_sequence(epochs_per_task: int = 30, lr: float = 5e-4,
                       backbone: str = "dinov2",
                       head: str = "multi",
                       scoring: str = "none",
                       batch_size: int = 64,
                       resume: bool = True,
                       checkpoint_path: str = None,
                       fisher_w1: float = 0.5,
                       fisher_w2: float = 0.5,
                       device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Train through all 10 tasks in sequence with replay buffer, active scoring,
    and automatic per-task checkpointing & resumption.
    """
    if checkpoint_path is None:
        checkpoint_path = f"checkpoint_phase3_{scoring}.pt"

    print(f"=== Full 10-task sequence ({epochs_per_task} epochs/task, backbone={backbone}, head={head}, scoring={scoring}) ===")
    train_loaders, val_loaders = get_task_loaders(backbone=backbone, batch_size=batch_size)
    model = HybridModel(backbone=backbone, head=head).to(device)

    # Initialize ReplayBuffer if scoring is enabled
    replay_buffer = ReplayBuffer(capacity=500, device=device) if scoring != "none" else None

    # acc_matrix[i][j] = accuracy on task i's val set after training through task j
    acc_matrix = [[None] * NUM_TASKS for _ in range(NUM_TASKS)]
    start_task = 0

    # Auto-resume check
    import os, json
    if resume and os.path.exists(checkpoint_path):
        print(f"\n>>> Found existing checkpoint at '{checkpoint_path}'. Loading state to resume... <<<")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        acc_matrix = ckpt["acc_matrix"]
        if replay_buffer is not None and ckpt.get("replay_buffer_state") is not None:
            replay_buffer.load_state_dict(ckpt["replay_buffer_state"])
        start_task = ckpt["completed_task"] + 1
        print(f">>> Resuming training from Task {start_task} (Tasks 0..{ckpt['completed_task']} already completed) <<<\n")

    for j in range(start_task, NUM_TASKS):
        print(f"\n--- Training task {j} ---")
        # Fresh optimizer per task
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_task)
        for epoch in range(epochs_per_task):
            train_one_task(model, train_loaders[j], optimizer, device, epochs=1, task_id=j,
                           replay_buffer=replay_buffer, replay_batch_size=50)
            scheduler.step()

        # Add candidates from Task j into replay buffer
        if scoring != "none":
            populate_buffer_from_task(model, train_loaders[j], replay_buffer, scoring=scoring,
                                      task_id=j, device=device, max_candidates=500,
                                      fisher_w1=fisher_w1, fisher_w2=fisher_w2)

        for i in range(j + 1):
            acc = evaluate_task(model, val_loaders[i], device, task_id=i)
            acc_matrix[i][j] = acc
            print(f"  after task {j}: task {i} accuracy = {acc * 100:.2f}%")

        # Save per-task checkpoint dictionary so progress is never lost
        ckpt_data = {
            "completed_task": j,
            "model_state_dict": model.state_dict(),
            "acc_matrix": acc_matrix,
            "replay_buffer_state": replay_buffer.state_dict() if replay_buffer else None,
        }
        torch.save(ckpt_data, checkpoint_path)
        
        # Also update intermediate JSON matrix and model weights
        intermediate_matrix_path = f"phase3_{scoring}_acc_matrix.json"
        intermediate_model_path = f"phase3_{scoring}_model.pt"
        with open(intermediate_matrix_path, "w") as f:
            json.dump(acc_matrix, f, indent=2)
        torch.save(model.state_dict(), intermediate_model_path)
        print(f"  [Checkpoint Saved] Task {j} saved to '{checkpoint_path}' & '{intermediate_matrix_path}'")

    return acc_matrix, model


if __name__ == "__main__":
    sanity_acc = run_single_task_sanity_check(epochs=30, backbone="dinov2", head="shared", scoring="none")
    if sanity_acc > 0.30:
        acc_matrix, model = run_full_sequence(epochs_per_task=30, backbone="dinov2", head="shared", scoring="none")
        torch.save(model.state_dict(), "phase3_none_model.pt")
        import json
        with open("phase3_none_acc_matrix.json", "w") as f:
            json.dump(acc_matrix, f, indent=2)
        print("Saved phase3_none_model.pt and phase3_none_acc_matrix.json")
