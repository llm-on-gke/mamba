import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch import optim
from torch.distributed import fsdp
from einops import rearrange, repeat

# ------------------------------------------------------------------------
# Pure PyTorch Mamba-2 (avoiding all Triton/CUDA custom ops)
# ------------------------------------------------------------------------

def chunk_state_ref(B, x, dt, dA_cumsum):
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    ngroups = B.shape[2]
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    
    # Decompose 4-operand einsum into TPU-safe pairwise operations
    decay_dt = decay_states.to(x.dtype) * dt.to(x.dtype)
    decay_dt = rearrange(decay_dt, "b h c l -> b c l h 1")
    B_decay = B.to(x.dtype) * decay_dt
    return torch.einsum("bclhn,bclhp->bchpn", B_decay, x)

def state_passing_ref(states, dA_chunk_cumsum, initial_states=None):
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
    dA_chunk_cumsum = torch.cat([torch.zeros_like(dA_chunk_cumsum[:, :, :1]), dA_chunk_cumsum], dim=-1)
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=torch.bool), diagonal=0)
    dt_chunk_segment_sum = torch.where(causal_mask, dt_chunk_segment_sum, -float('inf'))
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
    return out[:, :-1], out[:, -1]

def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    _, _, nchunks, chunk_size = dt.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool), diagonal=0)
    dt_segment_sum = torch.where(causal_mask, dt_segment_sum, -float('inf'))
    decay = torch.exp(dt_segment_sum)
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    
    # Decompose 3-operand einsum into TPU-safe pairwise operations
    x_reshaped = rearrange(x, "b (c s) h p -> b c s h p", c=nchunks)
    dt_x = rearrange(dt.to(x.dtype), "b h c s -> b c s h 1") * x_reshaped
    out = torch.einsum('bchls,bcshp->bclhp', scores_decay.to(x.dtype), dt_x)
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out if z is None else out * F.silu(z)

def ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False):
    batch, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[2]
    dstate = B.shape[-1]
    
    seqlen_padded = ((seqlen + chunk_size - 1) // chunk_size) * chunk_size
    pad_len = seqlen_padded - seqlen
    
    if pad_len > 0:
        dt = torch.cat([dt, torch.zeros(batch, pad_len, nheads, device=dt.device, dtype=dt.dtype)], dim=1)
        x = torch.cat([x, torch.zeros(batch, pad_len, nheads, headdim, device=x.device, dtype=x.dtype)], dim=1)
        B = torch.cat([B, torch.zeros(batch, pad_len, ngroups, dstate, device=B.device, dtype=B.dtype)], dim=1)
        C = torch.cat([C, torch.zeros(batch, pad_len, ngroups, dstate, device=C.device, dtype=C.dtype)], dim=1)
        if z is not None:
            z = torch.cat([z, torch.zeros(batch, pad_len, nheads, headdim, device=z.device, dtype=z.dtype)], dim=1)

    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    
    states = chunk_state_ref(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)
        
    states = rearrange(state_passing_ref(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                       "... (p n) -> ... p n", n=dstate)
    states = states.to(states_dtype)
    
    out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    
    if pad_len > 0:
        out = out[:, :seqlen, :, :]
    
    return out

class RMSNormGated(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x, z=None):
        if z is not None:
            x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Mamba2SimplePure(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // self.headdim
        self.chunk_size = chunk_size

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=True,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs
        )

        self.act = nn.SiLU()

        dt = torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))

        self.norm = RMSNormGated(self.d_inner, eps=1e-5).to(**factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def forward(self, u):
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        
        dt = F.softplus(dt + self.dt_bias)

        # 1D Convolution (causal by truncating padding)
        xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        xBC = xBC[:, :seqlen, :]

        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        
        y = ssd_chunk_scan_combined_ref(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            dt_bias=None,
            dt_softplus=False
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        y = self.norm(y, z)
        out = self.out_proj(y)
        return out


def worker_fn():
    # Initialize TPU distributed process group
    dist.init_process_group(backend='tpu_dist')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}/{world_size}] Worker started.")

    d_model = 128
    batch_size = 4
    seqlen = 64

    # Initialize pure PyTorch Mamba2 model on CPU
    model = Mamba2SimplePure(d_model=d_model, device='cpu')

    # Move the model to TPU *before* FSDP sharding to prevent incompatible tensor type errors.
    model = model.to('tpu')

    # Apply FSDP sharding to submodules and the main model
    fsdp.fully_shard(model.in_proj)
    fsdp.fully_shard(model.conv1d)
    fsdp.fully_shard(model.norm)
    fsdp.fully_shard(model.out_proj)
    fsdp.fully_shard(model)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create dummy data on TPU
    torch.manual_seed(42 + rank)
    x = torch.randn(batch_size, seqlen, d_model, device='tpu')
    y_target = torch.randn(batch_size, seqlen, d_model, device='tpu')

    print(f"[Rank {rank}] Starting training loop...")

    # Simple training loop
    for step in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.MSELoss()(out, y_target)
        loss.backward()
        optimizer.step()
        
        print(f"[Rank {rank}] Step {step} | Loss: {loss.item():.4f}")

    dist.destroy_process_group()
    print(f"[Rank {rank}] Training complete.")

if __name__ == '__main__':
    worker_fn()
