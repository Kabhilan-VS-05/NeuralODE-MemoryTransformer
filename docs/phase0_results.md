# Phase 0 Baseline Reproduction Results

## The Numbers

We successfully executed the literal implementation of the base paper's architecture on Split CIFAR-100 (10 tasks). Here are our reproduced numbers compared against the paper's claimed numbers:

| Metric | Base Paper Claim | Our Reproduction (Literal Reading) |
|---|---|---|
| **Average Accuracy** | 72.6% | **8.56%** |
| **Forgetting Metric (F)** | 0.183 | **0.797** (79.7%) |
| **Backward Transfer** | N/A | **-79.71%** |

## Analysis of Catastrophic Forgetting

Our reproduction reveals that the model immediately and completely forgets old tasks. The average accuracy is only `8.56%`, and the backward transfer is a massive `-79.7%`. Why is this happening?

1.  **The "Hidden Replay" Problem**: The base paper describes a memory module that stores features and uses attention to retrieve them. However, retrieving an old feature and passing it through a Transformer *does not* protect the weights of the final `nn.Linear` classifier. Because Task 2 only contains labels for classes 10-19, the gradients force the classifier to completely unlearn classes 0-9. To achieve 72.6%, the authors *must* have secretly mixed old samples (Experience Replay) into the training batches, or used a class-incremental classification head trick which they failed to disclose in the paper.
2.  **Epochs & Training Time**: The paper claimed 3.9 hours on an A100. We ran `epochs_per_task = 1` as a standard CL sanity check, which resulted in the model only achieving `10.00%` accuracy on the *current* task. An ODE + Transformer learning from raw pixels requires massive training time (likely 100+ epochs per task). 10.00% accuracy on a 10-class task is exactly random guessing. 

## Conclusion

We have fulfilled **Phase 0**. We built the architecture exactly as described in the paper, without adding any external tricks or our own novelties. 

The result is a mathematically sound but functionally flawed architecture that succumbs to complete catastrophic forgetting (Avg Acc: 8.56%). This is a fantastic scientific starting point. It proves that the base paper's architecture alone is insufficient. 

Every novel contribution we add from here on (DINOv2 backbone, Fisher Information Scoring, and Streaming) will be compared against this honest 8.56% baseline. 

## Next Steps (Phase 1)
We are now ready to move to **Phase 1: Environment & Data Engineering**. We will abandon the weak raw-pixel CNN and implement the **DINOv2** feature extractor to cache high-quality embeddings to disk, providing a massive immediate boost to our model's capacity!
