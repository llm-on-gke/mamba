import argparse
import faulthandler
import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import (
    PipelineStage,
    Schedule1F1B,
    ScheduleInterleavedZeroBubble,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity

import contextlib
import logging
from model_registry import ModelArgs

def get_logger():
    logger = logging.getLogger("playground")
    if not logger.handlers:
        ch = logging.StreamHandler()
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger

def rank_log(rank, logger, msg):
    if rank == 0:
        logger.info(f"[Rank {rank}] {msg}")
    else:
        print(f"[Rank {rank}] {msg}")

@contextlib.contextmanager
def profiler_range(name):
    yield

def profiler_range_push(name): pass
def profiler_range_pop(): pass

logger = get_logger()

class find_batch_size:
    @staticmethod
    def reset_memory(): pass
    @staticmethod
    def max_memory_on_any_rank(): return 0, 0
    @staticmethod
    def find_max_batch_size(model_name, runner_name, try_batch_size, sequence_length, work_dir, record_memory): return 2

def pipeline_parallel_model(model, pp_rank, pp_world_size, tp_size): return model

try:
    _rank = int(os.environ.get("RANK", "0"))
    _world_size = int(os.environ.get("WORLD_SIZE", "1"))
except:
    _rank = 0
    _world_size = 1

def compile_model(model):
    torch.compiler.reset()
    return torch.compile(model)

def get_model_config(model_name):
    if model_name in ["mamba2", "mamba3"]:
        from model_registry import get_model_config as registry_get
        return registry_get(model_name)

    if model_name == "1b":
        return ModelArgs(
            vocab_size=32768, device="cuda", multiple_of=512, dim=2048, n_layers=16
        )
    elif model_name == "7b":
        return ModelArgs(vocab_size=32768, device="cuda", multiple_of=512)
    elif model_name == "13b":
        return ModelArgs(
            vocab_size=32768,
            device="meta",
            multiple_of=512,
            dim=5120,
            n_heads=64,
            n_layers=40,
        )
    elif model_name == "70b":
        return ModelArgs(
            vocab_size=32768,
            device="meta",
            multiple_of=512,
            dim=8192,
            n_heads=64,
            n_layers=80,
        )
    elif model_name == "0.1b":
        return ModelArgs(
            vocab_size=32768,
            device="cuda",
            multiple_of=512,
            dim=1024,
            n_layers=4,
        )
    elif model_name == "short_wide":
        return ModelArgs(
            vocab_size=32768,
            device="meta",
            multiple_of=512,
            dim=16384,
            n_heads=64,
            n_layers=2,
        )
    elif model_name == "h100_dp":
        return ModelArgs(
            n_layers=12,
            vocab_size=32768,
            device="cuda",
            multiple_of=512,
        )
    elif model_name == "h100_dp_optim":
        return ModelArgs(
            n_layers=14,
            vocab_size=32768,
            device="cuda",
            multiple_of=512,
        )
    elif model_name == "mlp_only":
        return ModelArgs(
            multiple_of=512,
            dim=16384,
            n_heads=64,
            n_layers=2,
            vocab_size=32768,
            device="meta",
            mlp_only=True,
        )
    elif model_name == "simple_tp_test":
        class SimpleTpModelArgs:
            def __init__(self, device): pass
            def create_model(self, compile=False): return torch.nn.Linear(1,1)
        return SimpleTpModelArgs(device="cuda")
    else:
        raise ValueError(f"unknown model {model_name}")

def create_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=lr, fused=False)

class ModelRunner:
    def __init__(self, model_to_run, batch_size, sequence_length, compile, model_config=None):
        device = "cuda" if torch.cuda.is_available() else "tpu"
        self.model = model_to_run.create_model(compile)
        with profiler_range("init weights"):
            if hasattr(self.model, "init_weights"):
                self.model.init_weights()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.optimizer = create_optimizer(self.model)
        self.model_to_run = model_to_run
        self.model_config = model_config

    def run(self):
        self.optimizer.zero_grad(set_to_none=True)
        with profiler_range("generate random input"):
            assert self.batch_size > 0
            input_ids = create_input(self.batch_size, self.sequence_length, self.model_config)

        with profiler_range("forward"):
            with torch.autocast("cuda" if torch.cuda.is_available() else "tpu", torch.bfloat16):
                output = self.model(input_ids)
        with profiler_range("backward"):
            sum = output.sum()
            del output
            sum.backward()
        with profiler_range("optimizer"):
            self.optimizer.step()

def batch_sizes_to_run(max_batch_size):
    return sorted(
        {
            max(1, max_batch_size // 2),
            max(1, (max_batch_size * 3) // 4),
            max(1, (max_batch_size * 7) // 8),
        }
    )

@dataclass
class ModelToRun:
    name: str
    create_model: Callable[[], torch.nn.Module]
    dp_rank: int
    batch_size: int | None
    tensor_parallel: int
    model_config: object = None

    def create_model_runner(self, batch_size, sequence_length, compile):
        return ModelRunner(self, batch_size, sequence_length, compile, self.model_config)

    def with_new_batch_size(self, max_batch_size):
        return ModelToRun(
            self.name,
            self.create_model,
            self.dp_rank,
            max_batch_size,
            self.tensor_parallel,
            self.model_config
        )

    @property
    def batch_per_gpu_divisor(self):
        return self.tensor_parallel

    def batch_sizes_to_run(self):
        return batch_sizes_to_run(self.batch_size)

num_iterations = 6
lr = 3e-3
csv_prefix = "Output for csv:\n"

def create_input(batch_size, sequence_length, model_config=None):
    device = "cuda" if torch.cuda.is_available() else "tpu"
    if model_config and getattr(model_config, "name", "") in ["mamba2", "mamba3"]:
        d_model = getattr(model_config, "d_model", 2048)
        return torch.randn(batch_size, sequence_length, d_model, device=device)
    return torch.randint(32768, size=(batch_size, sequence_length), device=device)

def find_batch_size_for_model(
    model_name, model_to_run, sequence_length, work_dir, record_memory
):
    def try_batch_size(batch_size):
        runner = model_to_run.create_model_runner(
            batch_size, sequence_length, compile=False
        )

        for _ in range(3):
            runner.run()
        return runner.model, runner.optimizer

    max_batch_size = find_batch_size.find_max_batch_size(
        model_name,
        model_to_run.name,
        try_batch_size,
        sequence_length,
        work_dir,
        record_memory,
    )
    return model_to_run.with_new_batch_size(max_batch_size)

def profile_loop(
    name, batch_size, sequence_length, tensor_parallel, work_dir, pytorch_profiler
):
    rank_log(_rank, logger, f"{name} starting {batch_size=}")
    find_batch_size.reset_memory()
    
    start_iter_for_measurements = 2
    
    import time
    start_time = None
    
    for i in range(num_iterations):
        if i == start_iter_for_measurements:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if not pytorch_profiler and _rank == 0:
                    torch.cuda.cudart().cudaProfilerStart()
            start_time = time.time()
            profiler_range_push("main loop")
            
        with profiler_range(f"iteration {i}"):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            yield i
            peak_memory = find_batch_size.max_memory_on_any_rank()[1]
            rank_log(
                _rank,
                logger,
                f"{name} iter {i} complete, {batch_size=}, peak_memory={peak_memory}",
            )
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    profiler_range_pop()
    if not pytorch_profiler and _rank == 0 and torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStop()
        
    time_taken_seconds = end_time - start_time if start_time else 0.1
    batches_processed = (
        (num_iterations - start_iter_for_measurements) * batch_size / tensor_parallel
    )
    tokens_per_gpu_per_sec = sequence_length * batches_processed / time_taken_seconds
    sequences_per_second = batches_processed * _world_size / time_taken_seconds
    rank_log(
        _rank,
        logger,
        f"Ran combined batch size {batches_processed} in {time_taken_seconds:.2f} seconds. {tokens_per_gpu_per_sec:.2f} tokens/gpu/second, {sequences_per_second:.2f} sequences/second",
    )
    rank_log(
        _rank,
        logger,
        f"{csv_prefix}{name},{batch_size},{peak_memory},{tokens_per_gpu_per_sec:.2f},{sequences_per_second:.2f}",
    )

@contextmanager
def disable_gc2():
    thresholds = gc.get_threshold()
    gc.set_threshold(thresholds[0], thresholds[1], 1000000000)
    try:
        yield
    finally:
        gc.set_threshold(*thresholds)

def run_training(model_to_run, sequence_length, compile, work_dir, pytorch_profiler):
    find_batch_size.reset_memory()
    with disable_gc2():
        runner = model_to_run.create_model_runner(
            model_to_run.batch_size, sequence_length, compile=compile
        )
        for _i in profile_loop(
            model_to_run.name,
            model_to_run.batch_size,
            sequence_length,
            model_to_run.batch_per_gpu_divisor,
            work_dir=work_dir,
            pytorch_profiler=pytorch_profiler,
        ):
            runner.run()

def fully_shard_model(model, mesh: DeviceMesh | None = None, model_name=None):
    with profiler_range("fsdp model"):
        if model_name in ["mamba2", "mamba3"]:
            if model_name == "mamba2":
                fully_shard(model.in_proj, mesh=mesh)
                fully_shard(model.conv1d, mesh=mesh)
                fully_shard(model.norm, mesh=mesh)
                fully_shard(model.out_proj, mesh=mesh)
        else:
            if hasattr(model, 'layers'):
                for transformer_block in model.layers:
                    fully_shard(module=transformer_block, mesh=mesh)
    return fully_shard(model, mesh=mesh)

models = []
models_by_name = {}

def find_model(name):
    if name in models_by_name:
        return models_by_name[name]
    name_with_spaces = name.replace("_", " ")
    if name_with_spaces in models_by_name:
        return models_by_name[name_with_spaces]
    raise ValueError(
        f"Could not find config {name}. Valid options are: {models_by_name.keys()}"
    )

def add_model(model_to_run):
    models.append(model_to_run)
    models_by_name[model_to_run.name] = model_to_run

def register_models(mesh_1d, model_config, args):
    if args.run_dp or args.config is not None:
        def create_ddp_model(compile):
            device = "cuda" if torch.cuda.is_available() else "tpu"
            model = model_config.create_model().to(device)
            if compile:
                model = compile_model(model)
            result = DDP(model)
            def init_weights():
                model.init_weights()
            result.init_weights = init_weights
            return result

        add_model(
            ModelToRun(
                "dp",
                create_ddp_model,
                mesh_1d.get_rank() if mesh_1d else 0,
                batch_size=None,
                tensor_parallel=1,
                model_config=model_config
            )
        )
    if args.run_fsdp or args.config is not None:
        def create_fsdp_model(compile):
            device = "cuda" if torch.cuda.is_available() else "tpu"
            model = model_config.create_model().to(device)
            model_name = getattr(model_config, "name", None)
            model = fully_shard_model(model, mesh=mesh_1d, model_name=model_name)
            if compile:
                model = compile_model(model)
            return model

        add_model(
            ModelToRun(
                "fsdp",
                create_fsdp_model,
                mesh_1d.get_rank() if mesh_1d else 0,
                batch_size=None,
                tensor_parallel=1,
                model_config=model_config
            )
        )

def str_to_bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="+", required=False, default=None)
    parser.add_argument(
        "--batch-size", type=int, nargs="+", required=False, default=None
    )
    parser.add_argument("--sequence-length", type=int, required=False, default=512)
    parser.add_argument("--model", type=str, default="mamba2", required=False)
    parser.add_argument("--run-dp", type=str_to_bool, default=True, required=False)
    parser.add_argument("--run-fsdp", type=str_to_bool, default=True, required=False)
    parser.add_argument("--run-pp", type=str_to_bool, default=False, required=False)
    parser.add_argument("--work-dir", type=str, default=None, required=False)
    parser.add_argument(
        "--record-memory", type=str_to_bool, default=False, required=False
    )
    parser.add_argument(
        "--pytorch-profiler", type=str_to_bool, default=False, required=False
    )
    parser.add_argument("--compile", type=str_to_bool, default=False, required=False)
    args = parser.parse_args()

    try:
        device = torch.device("tpu")
        _ = torch.zeros(1, device=device)
        backend = "tpu_dist"
    except (RuntimeError, AttributeError, AssertionError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    
    mesh_1d = None

    model_config = get_model_config(args.model)
    register_models(mesh_1d, model_config, args)

    if args.config is not None:
        models_to_run = [find_model(config) for config in args.config]
    else:
        models_to_run = models

    for model_to_run in models_to_run:
        if args.batch_size is not None:
            batch_sizes = args.batch_size
        else:
            if model_to_run.batch_size is None:
                model_to_run = find_batch_size_for_model(
                    args.model,
                    model_to_run,
                    args.sequence_length,
                    work_dir=args.work_dir,
                    record_memory=False,
                )
            batch_sizes = model_to_run.batch_sizes_to_run()
            
        for batch_size in batch_sizes:
            with_batch_size = model_to_run.with_new_batch_size(batch_size)
            run_training(
                with_batch_size,
                args.sequence_length,
                compile=args.compile,
                work_dir=args.work_dir,
                pytorch_profiler=args.pytorch_profiler,
            )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()