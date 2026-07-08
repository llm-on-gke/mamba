import os
import sys
import torch
import torch.nn as nn
from torch import optim

# Ensure we can import from mamba_ssm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mamba_ssm.modules.mamba3 import Mamba3

def worker_fn():
    # Attempt to initialize torch_tpu
    try:
        from torch_tpu import api
        from torch import distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        device = api.tpu_device()
        dist.init_process_group(backend="tpu_dist")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        use_ddp = world_size > 1
    except ImportError:
        print("Running on CPU/GPU as torch_tpu is not available")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0
        world_size = 1
        use_ddp = False

    # Hyperparameters
    batch_size = 4
    seqlen = 128
    d_model = 256
    
    print(f"Rank {rank}/{world_size} initializing model on {device}...")
    
    # 1. Create Model
    model = Mamba3(
        d_model=d_model,
        d_state=16,
        expand=2,
        headdim=64,
        ngroups=1,
        device=device,
        dtype=torch.float32
    )

    if use_ddp:
        model = DDP(model)

    # 2. Create synthetic data
    # (batch_size, seqlen, d_model)
    torch.manual_seed(42 + rank)
    inputs = torch.randn(batch_size, seqlen, d_model, device=device)
    targets = torch.randn(batch_size, seqlen, d_model, device=device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"Rank {rank}: starting training steps...")
    for step in range(10):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Rank {rank}, step {step}, loss: {loss.item():.4f}")

    if use_ddp:
        dist.destroy_process_group()
        
    print(f"Rank {rank}: training complete.")

if __name__ == "__main__":
    worker_fn()
