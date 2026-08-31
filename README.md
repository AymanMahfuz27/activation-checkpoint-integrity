# Activation Checkpoint Integrity

Detect silent activation-checkpoint recomputation errors before they corrupt model updates.

## Problem

Activation checkpointing saves GPU memory by deleting forward-pass tensors and recomputing them during backpropagation. Random state, counters, mutable buffers, precision changes, or FP8 scaling can change recomputed values without changing tensor shape, type, or device. Metadata checks can miss this failure, so training may continue with incorrect gradients.

## Approach

The planned checker will create position-sensitive GPU fingerprints for original and recomputed saved tensors. Any mismatch will block the training step before model or optimizer state changes.

## Research scope

- Reproduce controlled same-metadata failures and matching correct cases.
- Verify saved-tensor coverage for every supported checkpoint pattern.
- Compare metadata checks, final-output hashing, tensor hashing, and exact comparison.
- Measure runtime, memory, kernel, synchronization, and compiler overhead.
- Evaluate pinned PyTorch and TorchTitan workloads in FP32 and BF16.

## Status

Research and implementation planning. Claims will follow reproducible measurements.
