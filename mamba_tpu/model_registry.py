# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

import math
import sys
import os
from dataclasses import dataclass
import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.modules.mamba2_simple import Mamba2Simple

@dataclass
class ModelArgs:
  dim: int = 256
  n_layers: int = 4
  n_heads: int = 8
  n_kv_heads: int | None = None
  vocab_size: int = 32000
  multiple_of: int = 256
  ffn_dim_multiplier: float | None = None
  norm_eps: float = 1e-5
  max_batch_size: int = 32
  max_seq_len: int = 2048
  device: str = "tpu"
  mlp_only: bool = False
  model_type: str = "llama2"

  def create_model(self, compile=False):
    if self.model_type == "mamba2simple":
      return Mamba2SimpleModel(self)
    if self.model_type == "mamba3":
      return Mamba3Model(self)
    return Llama2Model(self)

class FeedForward(nn.Module):
  def __init__(self, cfg: ModelArgs):
    super().__init__()
    # Calculate hidden dim matching multiple_of configuration
    hidden_dim = int(2 * (cfg.dim * 4) / 3)
    if cfg.ffn_dim_multiplier is not None:
      hidden_dim = int(cfg.ffn_dim_multiplier * hidden_dim)
    hidden_dim = cfg.multiple_of * ((hidden_dim + cfg.multiple_of - 1) // cfg.multiple_of)

    self.w1 = nn.Linear(cfg.dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, cfg.dim, bias=False)
    self.w3 = nn.Linear(cfg.dim, hidden_dim, bias=False)

  def forward(self, x):
    return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

def compute_rope_params(head_dim, theta_base=10000.0, context_length=2048, dtype=torch.float32):
  inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
  positions = torch.arange(context_length, dtype=dtype)
  angles = positions[:, None] * inv_freq[None, :]
  angles = torch.cat([angles, angles], dim=1)
  return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
  seq_len = x.shape[2]
  head_dim = x.shape[3]
  x1 = x[..., : head_dim // 2]
  x2 = x[..., head_dim // 2 :]
  cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
  sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
  rotated = torch.cat((-x2, x1), dim=-1)
  return (x * cos) + (rotated * sin)

class GroupedQueryAttention(nn.Module):
  def __init__(self, cfg: ModelArgs):
    super().__init__()
    self.n_heads = cfg.n_heads
    self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads
    self.head_dim = cfg.dim // cfg.n_heads
    self.group_size = self.n_heads // self.n_kv_heads

    self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
    self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

  def forward(self, x, mask, cos, sin):
    b, num_tokens, _ = x.shape
    queries = self.wq(x).view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
    keys = self.wk(x).view(b, num_tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
    values = self.wv(x).view(b, num_tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)

    queries = apply_rope(queries, cos, sin)
    keys = apply_rope(keys, cos, sin)

    keys = keys.repeat_interleave(self.group_size, dim=1)
    values = values.repeat_interleave(self.group_size, dim=1)

    attn_scores = queries @ keys.transpose(2, 3)
    attn_scores = attn_scores.masked_fill(mask, -torch.inf)
    attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)

    context_vec = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, -1)
    return self.wo(context_vec)

class TransformerBlock(nn.Module):
  def __init__(self, cfg: ModelArgs):
    super().__init__()
    self.attention = GroupedQueryAttention(cfg)
    self.feed_forward = FeedForward(cfg)
    self.attention_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
    self.ffn_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)

  def forward(self, x, mask, cos, sin):
    x = x + self.attention(self.attention_norm(x), mask, cos, sin)
    x = x + self.feed_forward(self.ffn_norm(x))
    return x

class Llama2Model(nn.Module):
  def __init__(self, cfg: ModelArgs):
    super().__init__()
    self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
    self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
    self.norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
    self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    cos, sin = compute_rope_params(
        head_dim=cfg.dim // cfg.n_heads,
        context_length=cfg.max_seq_len,
    )
    self.register_buffer("cos", cos, persistent=False)
    self.register_buffer("sin", sin, persistent=False)
    self.cfg = cfg

  def init_weights(self):
    def _init_weights(m):
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
      elif isinstance(m, nn.RMSNorm):
        nn.init.ones_(m.weight)
    self.apply(_init_weights)

  def forward(self, in_idx):
    x = self.tok_embeddings(in_idx)
    num_tokens = x.shape[1]
    mask = torch.triu(
        torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
        diagonal=1,
    )
    for block in self.layers:
      x = block(x, mask, self.cos, self.sin)
    x = self.norm(x)
    return self.output(x)

class Mamba3Block(nn.Module):
  def __init__(self, cfg: ModelArgs):
    super().__init__()
    # Use config values; defaulting some Mamba3 specific args.
    self.mamba = Mamba3(d_model=cfg.dim, device=cfg.device)
    self.mamba_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
    self.feed_forward = FeedForward(cfg)
    self.ffn_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)

  def forward(self, x):
    # Mamba3 kernel doesn't take mask, cos, sin like GroupedQueryAttention does
    x = x + self.mamba(self.mamba_norm(x))
    x = x + self.feed_forward(self.ffn_norm(x))
    return x

class Mamba3Model(nn.Module):
  def __init__(self, cfg: ModelArgs):
    super().__init__()
    self.layers = nn.ModuleList([Mamba3Block(cfg) for _ in range(cfg.n_layers)])
    self.norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
    self.cfg = cfg

  def init_weights(self):
    # mamba3 may not need LLM related initialization
    pass

  def forward(self, x):
    for block in self.layers:
      x = block(x)
    x = self.norm(x)
    return x

class Mamba2SimpleBlock(nn.Module):
  def __init__(self, cfg: ModelArgs):
    super().__init__()
    # Use config values; defaulting some Mamba2 specific args.
    self.mamba = Mamba2Simple(d_model=cfg.dim, device=cfg.device)
    self.mamba_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
    self.feed_forward = FeedForward(cfg)
    self.ffn_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)

  def forward(self, x):
    x = x + self.mamba(self.mamba_norm(x))
    x = x + self.feed_forward(self.ffn_norm(x))
    return x

class Mamba2SimpleModel(nn.Module):
  def __init__(self, cfg: ModelArgs):
    super().__init__()
    self.layers = nn.ModuleList([Mamba2SimpleBlock(cfg) for _ in range(cfg.n_layers)])
    self.norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
    self.cfg = cfg

  def init_weights(self):
    pass

  def forward(self, x):
    for block in self.layers:
      x = block(x)
    x = self.norm(x)
    return x