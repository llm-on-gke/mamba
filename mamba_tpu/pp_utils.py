# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

def pipeline_parallel_model(model, rank, interleaved_size, tp_size):
  # Mock pipeline parallel wrapper, returns model unchanged.
  return model
