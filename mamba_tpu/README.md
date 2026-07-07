# Mamba2 on TPU (Ironwood)

This directory contains code and instructions to run the `Mamba2Simple` model on a Google Kubernetes Engine (GKE) cluster equipped with TPUs (Ironwood 2x2x4 shape, effectively `v5litepod-16`).

## Files

- `train.py`: A PyTorch script utilizing `torch_tpu` to train the `Mamba2Simple` model on TPU using PyTorch Distributed Data Parallel (DDP). It generates synthetic data for inputs and targets.

## Model Changes

The original `mamba_ssm/modules/mamba2_simple.py` used `mamba_chunk_scan_combined` from Triton which is not directly compatible with TPU. We've rewritten it to use a standard PyTorch fallback built upon `ssd_minimal_discrete` logic while preserving the exact operations, which makes it natively compatible with `torch_tpu` and TPU accelerators.

## How to Run on GKE

1. **Build a Docker image**: Ensure your Docker image includes `torch_tpu`, `einops`, and other dependencies.
2. **Push the image**: Push it to your Google Artifact Registry (GAR).
3. **Deploy the Job**: Apply a Kubernetes Job manifest to your GKE cluster configured with the Ironwood TPU node pool (`v5litepod-16` topology).

A sample Job configuration snippet:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: mamba2-tpu-training
spec:
  template:
    spec:
      containers:
      - name: mamba2-trainer
        image: <YOUR_GAR_IMAGE_PATH>
        command: ["python3", "mamba_tpu/train.py"]
        resources:
          limits:
            google.com/tpu: "16" # For Ironwood 2x2x4 (v5e-16)
      nodeSelector:
        cloud.google.com/gke-tpu-topology: "2x2x4"
        cloud.google.com/gke-tpu-accelerator: "tpu-v5-lite-podslice"
      restartPolicy: Never
```
