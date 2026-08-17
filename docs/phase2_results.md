# Phase 1 & 2 Results

Before moving to the memory scoring in Phase 3, we implemented two major architectural updates to the baseline in Phase 1 and 2. Here are the documented results of those shifts.

## Phase 1: DINOv2 Backbone (Task-Incremental)
In Phase 1, we replaced the weak CNN with frozen **DINOv2** features. We retained the multi-head classifier structure (where the model knows the task ID at test time).

* **Average Accuracy:** **94.16%**
* **Forgetting Metric (F):** **0.035** (3.5%)
* **Backward Transfer:** **-3.48%**

**Analysis:** DINOv2 provides incredibly robust, linearly separable features. Because the multi-head setup knows the task ID at inference (preventing cross-task output space collisions), catastrophic forgetting is almost entirely eliminated, far exceeding the paper's 72.6% claim. However, this is "Task-Incremental Learning," which is an easier problem than true Class-Incremental Learning.

## Phase 2: Shared Head (Class-Incremental)
In Phase 2, we moved to the much harder **Class-Incremental** setup by replacing the multi-head with a single **Shared Head**. The model no longer receives the task ID at inference and must map all 100 classes into a single output space.

* **Average Accuracy:** **10.76%**
* **Forgetting Metric (F):** **0.963** (96.3%)
* **Backward Transfer:** **-96.27%**

**Analysis:** This is catastrophic forgetting in its purest form. Because we have no replay mechanism yet, training on Task 2 (classes 10-19) immediately overwrites the shared weights for Task 1 (classes 0-9). The model only remembers the very last task it was trained on (10 classes out of 100 = 10% accuracy). 

This 10.76% baseline is exactly what we used in Phase 3 to measure the effectiveness of the Random, Influence, and Fisher memory scoring techniques.
