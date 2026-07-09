# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import gc

def reset_memory():
  gc.collect()
  if hasattr(torch, "tpu"):
    torch.tpu.synchronize()

def max_memory_on_any_rank():
  # TPU doesn't support CUDA-specific memory query directly via this namespace.
  # We return placeholder stats for logging.
  return 0, 0

def tensor_memory(p):
  return p.numel() * p.element_size()

def find_max_batch_size(model_name, run_name, try_batch_size, sequence_length, work_dir, record_memory):
  # Sweeps batch sizes 1, 2, 4, 8, 16, 32.
  # Stops if try_batch_size raises a memory exception.
  current_batch = 1
  last_working = 1
  try:
    while current_batch <= 32:
      try:
        try_batch_size(current_batch)
        last_working = current_batch
        current_batch *= 2
      except RuntimeError as e:
        print(f"Batch size {current_batch} failed or hit OOM: {e}")
        break
  except Exception as e:
    print(f"Exception during find_max_batch_size: {e}")
  return last_working
