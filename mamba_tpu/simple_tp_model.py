# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
from torch import nn

class ModelArgs:
  def __init__(self, device="tpu"):
    self.device = device
    self.n_layers = 2
    self.vocab_size = 32000
    self.dim = 256
    self.n_heads = 8

  def create_model(self):
    class SimpleModel(nn.Module):
      def __init__(self):
        super().__init__()
        self.tok_embeddings_simple = nn.Embedding(32000, 256)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention_norm": nn.RMSNorm(256),
                "attention": nn.ModuleDict({
                    "wq": nn.Linear(256, 256, bias=False),
                    "wk": nn.Linear(256, 256, bias=False),
                    "wv": nn.Linear(256, 256, bias=False),
                    "wo": nn.Linear(256, 256, bias=False),
                }),
                "ffn_norm": nn.RMSNorm(256),
                "feed_forward": nn.ModuleDict({
                    "w1": nn.Linear(256, 1024, bias=False),
                    "w2": nn.Linear(1024, 256, bias=False),
                    "w3": nn.Linear(256, 1024, bias=False),
                })
            }) for _ in range(2)
        ])
        self.norm_simple = nn.RMSNorm(256)
        self.output_simple = nn.Linear(256, 32000, bias=False)

      def init_weights(self):
        pass

      def forward(self, x):
        h = self.tok_embeddings_simple(x)
        for layer in self.layers:
          h_norm = layer["attention_norm"](h)
          q = layer["attention"]["wq"](h_norm)
          k = layer["attention"]["wk"](h_norm)
          v = layer["attention"]["wv"](h_norm)
          attn = q @ k.transpose(-2, -1)
          attn = torch.softmax(attn / 16.0, dim=-1)
          context = attn @ v
          h = h + layer["attention"]["wo"](context)
          h_ff = layer["ffn_norm"](h)
          ff = nn.functional.silu(layer["feed_forward"]["w1"](h_ff)) * layer["feed_forward"]["w3"](h_ff)
          h = h + layer["feed_forward"]["w2"](ff)
        return self.output_simple(self.norm_simple(h))
    return SimpleModel()
