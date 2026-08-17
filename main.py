"""
main.py
-------
Entry point for Phase 0: literal base-paper replication.

Usage:
    # 1. Local, CPU/4GB-GPU dummy-tensor sanity checks (no real data/training):
    python models/ode.py
    python models/memory.py
    python models/transformer.py
    python models/hybrid.py
    python training/influence_scoring.py

    # 2. Local, real data, quick sanity (small epoch count, verifies no crashes):
    python main.py --mode sanity --epochs 2

    # 3. Colab/Kaggle, real run:
    python main.py --mode sanity --epochs 30      # diagnostic 1, must pass first
    python main.py --mode full --epochs 30        # diagnostic 2 / real Phase 0 result
    python evaluate.py                             # prints final metrics
"""

import argparse
import json
import torch

from training.engine_baseline import run_single_task_sanity_check, run_full_sequence
from training.engine_streaming import run_streaming_sequence
from data_cached import build_streaming_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sanity", "full", "streaming"], default="sanity")
    parser.add_argument("--backbone", choices=["cnn", "dinov2"], default="dinov2",
                        help="Backbone feature extractor: 'dinov2' (cached) or 'cnn' (raw images)")
    parser.add_argument("--head", choices=["multi", "shared"], default="shared",
                        help="Classifier head: 'multi' (10 separate 10-way heads) or 'shared' (single 100-way head)")
    parser.add_argument("--scoring", choices=["none", "random", "influence", "fisher", "fisher_proto"], default="none",
                        help="Replay candidate scoring method: 'none', 'random', 'influence', 'fisher', 'fisher_proto'")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Automatically resume from last completed task if checkpoint exists")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Disable resuming and start from Task 0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fisher_w1", type=float, default=0.5,
                        help="Weight for Fisher sensitivity in fisher_proto scoring")
    parser.add_argument("--fisher_w2", type=float, default=0.5,
                        help="Weight for prototype distance penalty in fisher_proto scoring")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Selected: Backbone={args.backbone.upper()} | Head={args.head.upper()} | Scoring={args.scoring.upper()} | Resume={args.resume}")

    if args.mode == "sanity":
        acc = run_single_task_sanity_check(
            epochs=args.epochs, lr=args.lr, backbone=args.backbone, head=args.head,
            scoring=args.scoring, device=device)
        if acc <= 0.30:
            print("\nDo NOT proceed to --mode full yet -- fix the architecture/training "
                  "regime first (see the FAIL message above).")
    elif args.mode == "streaming":
        # We reuse the args.epochs parameter as epochs_per_stream
        batch_size = 64
        streaming_loader, val_loaders, total_batches = build_streaming_loader(
            batch_size=batch_size, epochs_per_stream=args.epochs
        )
        acc_history, model = run_streaming_sequence(
            streaming_loader=streaming_loader,
            val_loaders=val_loaders,
            total_batches=total_batches,
            lr=args.lr, backbone=args.backbone, head=args.head,
            scoring=args.scoring, fisher_w1=args.fisher_w1, fisher_w2=args.fisher_w2, device=device
        )
        
        model_save_path = f"phase4_{args.scoring}_streaming_model.pt"
        matrix_save_path = f"phase4_{args.scoring}_streaming_history.json"

        torch.save(model.state_dict(), model_save_path)
        with open(matrix_save_path, "w") as f:
            json.dump(acc_history, f, indent=2)
        print(f"\nSaved {model_save_path} and {matrix_save_path}")
    else:
        acc_matrix, model = run_full_sequence(
            epochs_per_task=args.epochs, lr=args.lr, backbone=args.backbone, head=args.head,
            scoring=args.scoring, resume=args.resume, fisher_w1=args.fisher_w1, fisher_w2=args.fisher_w2, device=device)
        
        model_save_path = f"phase3_{args.scoring}_model.pt"
        matrix_save_path = f"phase3_{args.scoring}_acc_matrix.json"

        torch.save(model.state_dict(), model_save_path)
        with open(matrix_save_path, "w") as f:
            json.dump(acc_matrix, f, indent=2)
        print(f"\nSaved {model_save_path} and {matrix_save_path}")
        print(f"Run `python evaluate.py --matrix {matrix_save_path}` to see the final metrics.")


if __name__ == "__main__":
    main()
