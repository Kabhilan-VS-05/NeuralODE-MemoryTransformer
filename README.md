# Neural ODE Memory Transformer (Continual Learning R&D)

This repository contains the ongoing R&D codebase for benchmarking and advancing memory retention mechanisms in Continual Learning environments. We are currently evaluating various replay buffer scoring mechanisms on CIFAR-100 using DINOv2 extracted features.

## Project Phases

This codebase captures our progression through the following R&D phases:

*   **Phase 0-2 (Offline Baselines):** Establishing a rigid task-based continual learning loop (`run_full_sequence`) using pre-extracted DINOv2 features. 
*   **Phase 3 (Memory Scoring Benchmarks):** We benchmarked several 500-sample eviction strategies to combat catastrophic forgetting:
    *   **Random:** Naive random uniform selection.
    *   **Influence Functions:** Pure gradient-magnitude based selection.
    *   **Fisher:** Our custom Fisher Information implementation.
    *   **Fisher-Proto (Ours):** A hybrid scoring mechanism that blends Fisher gradient sensitivity (identifying hard boundaries) with a Prototype-Distance penalty (enforcing class representativeness and punishing noisy outliers).
    *   *Note: Detailed quantitative results and analysis for all offline benchmarks can be found in the `docs/` directory.*
*   **Phase 4 (Streaming Continual Learning):** Transitioning from artificial, discrete task boundaries to true, continuous data streams (`run_streaming_sequence`). This includes a dynamic "micro-buffer" architecture capable of iteratively scoring and flushing memory on the fly.

## Core Architecture

*   **`main.py`**: The central entry point. Supports running single-task sanity checks (`--mode sanity`), the rigid 10-task offline sequence (`--mode full`), and the continuous data stream (`--mode streaming`).
*   **`models/`**: Contains the core `HybridModel` which maps 384-dimensional DINOv2 embeddings to a `shared` 100-way or `multi` 10-way classification head.
*   **`training/engine_baseline.py`**: The traditional offline task-by-task training loop.
*   **`training/engine_streaming.py`**: The continuous streaming engine, featuring micro-buffer aggregation and periodic memory flushing.
*   **`training/fisher_scoring.py`**: Implements the logic for our custom tunable `fisher_proto` memory selection scoring.
*   **`data_cached.py`**: Handles loading pre-computed DINOv2 tensors to bypass heavy CNN feature extraction during rapid R&D iteration. Includes `build_streaming_loader()` for infinite continuous data simulation.
*   **`docs/`**: Historical markdown artifacts detailing empirical results, analysis, and implementation plans across phases.

## Quick Start

**1. Generate Cached DINOv2 Features:**
Before training, you must extract and cache the CIFAR-100 dataset features to disk:
```bash
python data_dinov2.py
```

**2. Run Offline Benchmark Sequence:**
Run a 10-task sequence using our custom Fisher-Proto memory scoring mechanism:
```bash
python main.py --mode full --backbone dinov2 --head shared --scoring fisher_proto --fisher_w1 0.3 --fisher_w2 0.7 --epochs 30
```

**3. Run Streaming Evaluation:**
Run a continuous data stream, dynamically flushing memory via random selection:
```bash
python main.py --mode streaming --backbone dinov2 --head shared --scoring random --epochs 1
```
