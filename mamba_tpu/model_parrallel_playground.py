import argparse
import time
import faulthandler
import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch
import torch_tpu  # Required to enable TPU backend support in PyTorch
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

import find_batch_size
import simple_tp_model
from model_registry import ModelArgs
from log_utils import (
    get_logger,
    profiler_range,
    profiler_range_pop,
    profiler_range_push,
    rank_log,
)
from pp_utils import pipeline_parallel_model

"""
This is adapted from the pytorch fsdp_tp_example.py but modified to test also test
pipeline parallelism. It also fixes a few performance issues to make the results more
representative. Nothing in here is IP but all of this is still very much (c) Jane Street.
Meaning if you find this file somewhere, don't just copy it.

To run this build the ocaml wrapper in bin and then run that e.g.

./bin/run_example.exe reno -bid-acquire 1000 -bid-finish 20000 -cluster skadi_bench -num-gpus 8 -output-dir /j/rs1/app/hive/user-/non-intern/mskarupke/2025/benchmark_parallelism/ -model h100_dp -config fsdp

Roughly the behavior is this:
- If you don't pass anything, it will run a full sweep over all configurations. Don't do
  this unless you have a lot of time.
- If you pass a config (or multiple configs) it'll run those configs. E.g. "fsdp" or
  "pipeline_parallel_2x_pipelinelength=4"
- If you pass a batch-size (or multiple) it won't run a sweep, it'll just run that batch
  size.
"""

logger = get_logger()

# understand world topology
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])


def compile_model(model):
    # Unfortunately we need to reset the compiler between different models. Otherwise we
    # get the most mysterious errors.
    torch.compiler.reset()
    return torch.compile(model)


def get_model_config(model_name):
    if model_name == "1b":
        return ModelArgs(
            vocab_size=32768, device="tpu", multiple_of=512, dim=2048, n_layers=16
        )
    elif model_name == "7b":
        return ModelArgs(vocab_size=32768, device="tpu", multiple_of=512)
    elif model_name == "mamba3_7b":
        return ModelArgs(vocab_size=32768, device="tpu", multiple_of=512, model_type="mamba3")
    elif model_name == "mamba2simple_7b":
        return ModelArgs(vocab_size=32768, device="tpu", multiple_of=512, model_type="mamba2simple")
    elif model_name == "13b":
        # arguments from https://www.reddit.com/r/LocalLLaMA/comments/15514s1/why_7_13_30b/
        return ModelArgs(
            vocab_size=32768,
            device="meta",
            multiple_of=512,
            dim=5120,
            n_heads=64,  # mskarupke: this is wrong. should be 40. But needs to be 64 to allow tensor-parallel 64
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
        # small model for quick iterations when testing code changes
        return ModelArgs(
            vocab_size=32768,
            device="tpu",
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
            device="tpu",
            multiple_of=512,
        )
    elif model_name == "h100_dp_optim":
        return ModelArgs(
            n_layers=14,
            vocab_size=32768,
            device="tpu",
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
        return simple_tp_model.ModelArgs(device="tpu")
    else:
        raise ValueError(f"unknown model {model_name}")


def create_optimizer(model):
    # Fused optimizers are not implemented for TPU (and not needed, as XLA fuses them anyway)
    return torch.optim.AdamW(model.parameters(), lr=lr, fused=False)


class ModelRunner:
    def __init__(self, model_to_run, batch_size, sequence_length, compile):
        self.model = model_to_run.create_model(compile).to_empty(device="tpu")
        with profiler_range("init weights"):
            self.model.init_weights()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.optimizer = create_optimizer(self.model)
        self.model_to_run = model_to_run

    def run(self):
        self.optimizer.zero_grad(set_to_none=True)
        with profiler_range("generate random input"):
            assert self.batch_size > 0
            if hasattr(self.model, 'tok_embeddings') or hasattr(self.model, 'module') and hasattr(self.model.module, 'tok_embeddings'):
                input_ids = create_input(self.batch_size, self.sequence_length)
            else:
                dim = getattr(self.model, 'cfg', getattr(self.model, 'module', self.model).cfg).dim
                input_ids = torch.randn(self.batch_size, self.sequence_length, dim, device="tpu")

        with profiler_range("forward"):
            # Make sure to use bf16 to run on tensor cores. Otherwise the networking is
            # a small fraction of the overall cost, since the matrix multiplies are so
            # slow when not using tensor cores.
            with torch.autocast("tpu", torch.bfloat16):
                output = self.model(input_ids)
        with profiler_range("backward"):
            sum = output.sum()
            del output
            sum.backward()
        with profiler_range("optimizer"):
            self.optimizer.step()


def batch_sizes_to_run(max_batch_size):
    # Run 0.5*max, 0.75*max and 0.875*max. Have them in a set to deduplicate because for
    # small numbers there will be duplicates here
    return sorted(
        {
            max(1, max_batch_size // 2),
            max(1, (max_batch_size * 3) // 4),
            max(1, (max_batch_size * 7) // 8),
            # max_batch_size,
        }
    )


@dataclass
class ModelToRun:
    name: str
    create_model: Callable[[], torch.nn.Module]
    dp_rank: int
    batch_size: int | None
    tensor_parallel: int

    def create_model_runner(self, batch_size, sequence_length, compile):
        return ModelRunner(self, batch_size, sequence_length, compile)

    def with_new_batch_size(self, max_batch_size):
        return ModelToRun(
            self.name,
            self.create_model,
            self.dp_rank,
            max_batch_size,
            self.tensor_parallel,
        )

    @property
    def batch_per_gpu_divisor(self):
        return self.tensor_parallel

    def batch_sizes_to_run(self):
        return batch_sizes_to_run(self.batch_size)


num_iterations = 6
lr = 3e-3
csv_prefix = "Output for csv:\n"


def create_input(batch_size, sequence_length):
    return torch.randint(32768, size=(batch_size, sequence_length), device="tpu")


def find_batch_size_for_model(
    model_name, model_to_run, sequence_length, work_dir, record_memory
):
    def try_batch_size(batch_size):
        # When running a model as part of [try_batch_size], don't compile the model. We
        # want to quickly run a sweep in this case and compiling just takes too
        # long. Especially when the ojb has to restart because of hangs, which happens
        # often on OOM.
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
    start_time = 0.0
    end_time = 0.0
    start_iter_for_measurements = 2
    for i in range(num_iterations):
        if i == start_iter_for_measurements:
            torch.tpu.synchronize()
            if pytorch_profiler:
                profiler_cm = torch.profiler.profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XLA],
                    record_shapes=True,
                )
                # start and stop once because there is a bug in pytorch 2.9
                profiler_cm.__enter__()
                profiler_cm.__exit__(None, None, None)
                local_profiler = profiler_cm.__enter__()
            start_time = time.perf_counter()

            # push a separate range after the first iteration, to measure just the time of
            # a training loop, without the overhead of the first iter
            profiler_range_push("main loop")
        with profiler_range(f"iteration {i}"):
            # reset_peak_memory_stats is not implemented on TPU
            pass
            yield i
            peak_memory = find_batch_size.max_memory_on_any_rank()[1]
            rank_log(
                _rank,
                logger,
                f"{name} iter {i} complete, {batch_size=}, peak_memory={peak_memory}",
            )
    torch.tpu.synchronize()
    end_time = time.perf_counter()
    profiler_range_pop()
    if pytorch_profiler:
        profiler_cm.__exit__(None, None, None)
        local_profiler.export_chrome_trace(f"{work_dir}/profile{_rank}.json")
    time_taken_seconds = end_time - start_time
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
    # turn off gc.collect(2)
    gc.set_threshold(thresholds[0], thresholds[1], 1000000000)
    try:
        yield
    finally:
        gc.set_threshold(*thresholds)


def run_training(model_to_run, sequence_length, compile, work_dir, pytorch_profiler):
    # Need to always reset the torch allocator before switching models. It likes to OOM
    # early with e.g. 58gb in use and 14gb "reserved by PyTorch but unallocated." Meaning
    # we OOM with 14gb unused. Probably because of fragmentation. You can modify
    # PYTORCH_CUDA_ALLOC_CONF to have less of this, but this would sometimes lead to
    # noticeable slowdowns in the benchmarks, similar in character to the overhead caused
    # by the python GC. (meaning one GPU is busy with memory management that takes
    # hundreds of milliseconds while 63 GPUs are waiting for it) So after trying a few
    # things, resetting between runs seems like the best option.
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


def fully_shard_model(model, mesh: DeviceMesh | None = None):
    with profiler_range("fsdp model"):
        for transformer_block in model.layers:
            fully_shard(module=transformer_block, mesh=mesh)
    return fully_shard(model, mesh=mesh)


def tensor_parallel_model(model, mesh: DeviceMesh):
    with profiler_range("parallelize model"):
        # parallelize the first embedding and the last linear out projection
        tp_model = parallelize_module(
            model,
            mesh,
            {
                "tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                    # use_local_output=False,
                ),
                "norm": SequenceParallel(),
                "output": ColwiseParallel(
                    input_layouts=Shard(1), output_layouts=Replicate()
                ),
                # the below are for the "simple_tp_test" model
                "tok_embeddings_simple": ColwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                    use_local_output=False,
                ),
                "output_simple": ColwiseParallel(
                    input_layouts=Shard(1), output_layouts=Replicate()
                ),
            },
        )

        for transformer_block in tp_model.layers:  # type: ignore
            layer_tp_plan = {
                "attention_norm": SequenceParallel(),
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), None),  # type: ignore
                    desired_input_layouts=(Replicate(), None),  # type: ignore
                ),
                "attention.wq": ColwiseParallel(),
                "attention.wk": ColwiseParallel(),
                "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
                "ffn_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                "feed_forward.w3": ColwiseParallel(),
                "linear0": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Shard(-1),
                    use_local_output=False,
                ),
                "linear1": RowwiseParallel(
                    input_layouts=Shard(-1),
                    output_layouts=Shard(1),
                    use_local_output=False,
                ),
                "norm_simple": SequenceParallel(),
            }

            # Adjust attention module to use the local number of heads
            if hasattr(transformer_block, "attention"):
                attn_layer = transformer_block.attention
                assert (attn_layer.n_heads % mesh.size()) == 0
                attn_layer.n_heads = attn_layer.n_heads // mesh.size()
                attn_layer.n_kv_heads = attn_layer.n_kv_heads // mesh.size()

            # Custom parallelization plan for the model
            parallelize_module(
                module=transformer_block,
                device_mesh=mesh,
                parallelize_plan=layer_tp_plan,
            )
        return tp_model


def tensor_parallel_model_ulysses(model, mesh: DeviceMesh):
    with profiler_range("parallelize model"):
        shard_hidden = Shard(-1)
        shard_sequence = Shard(1)
        switch_to_shard_sequence = PrepareModuleInput(
            input_layouts=Replicate(),
            desired_input_layouts=shard_sequence,
            use_local_output=True,
        )
        switch_to_replicate = PrepareModuleOutput(
            output_layouts=(shard_sequence,), desired_output_layouts=(Replicate(),)
        )
        # parallelize the first embedding and the last linear out projection
        tp_model = parallelize_module(
            model,
            mesh,
            {
                "tok_embeddings": switch_to_shard_sequence,
                # "norm": SequenceParallel(),
                "output": switch_to_replicate,
                # the below are for the "simple_tp_test" model
                "tok_embeddings_simple": switch_to_shard_sequence,
                "output_simple": switch_to_replicate,
            },
        )

        for transformer_block in tp_model.layers:  # type: ignore
            switch_to_shard_hidden = PrepareModuleOutput(
                output_layouts=shard_sequence, desired_output_layouts=shard_hidden
            )
            layer_tp_plan = {
                "attention.wq": switch_to_shard_hidden,
                "attention.wk": switch_to_shard_hidden,
                "attention.wv": switch_to_shard_hidden,
                "attention.wo": PrepareModuleInput(
                    input_layouts=shard_hidden,
                    desired_input_layouts=shard_sequence,
                    use_local_output=True,
                ),
            }

            # Adjust attention module to use the local number of heads
            if hasattr(transformer_block, "attention"):
                attn_layer = transformer_block.attention
                assert (attn_layer.n_heads % mesh.size()) == 0
                attn_layer.n_heads = attn_layer.n_heads // mesh.size()
                attn_layer.n_kv_heads = attn_layer.n_kv_heads // mesh.size()

            # Custom parallelization plan for the model
            parallelize_module(
                module=transformer_block,
                device_mesh=mesh,
                parallelize_plan=layer_tp_plan,
            )
        return tp_model


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


def pipeline_parallel_name(pp_size, pipelinelength, tp_size, interleave):
    result = f"pipeline parallel {pp_size}x {pipelinelength=}"
    if tp_size > 1:
        result += f" tp={tp_size}"
    if interleave > 1:
        result += f" {interleave=}"
    return result


class PipelineParallelRunner:
    def __init__(
        self,
        pp_size,
        per_gpu_batch_size,
        sequence_length,
        pipeline_length,
        create_model,
        tp_size,
        interleave,
    ):
        assert _world_size % (pp_size * interleave * tp_size) == 0
        self.pp_size = pp_size
        self.interleave = interleave
        self.pipeline_length = pipeline_length
        dp_size = _world_size // pp_size // tp_size
        device_mesh = init_device_mesh(
            "tpu", (dp_size, pp_size, tp_size), mesh_dim_names=("dp", "pp", "tp")
        )
        self.pp_mesh = device_mesh["pp"]
        self.pp_rank = self.pp_mesh.get_local_rank()
        self.batch_size = per_gpu_batch_size * pipeline_length
        self.sequence_length = sequence_length
        self.pp_world_size = self.pp_mesh.size()
        if interleave == 1:
            model = pipeline_parallel_model(
                create_model(), self.pp_rank, self.pp_world_size, tp_size
            )

            if tp_size > 1:
                model = tensor_parallel_model(model, device_mesh["tp"])
            model = model.to_empty(device="tpu")
            with profiler_range("init weights"):
                model.init_weights()

            self.optimizer = create_optimizer(model)
            self.model = model
        else:
            interleaved_size = self.pp_world_size * interleave
            self.optimizers = []
            self.models = []
            for i in range(interleave):
                rank = self.pp_rank + i * self.pp_world_size

                model = pipeline_parallel_model(
                    create_model(), rank, interleaved_size, tp_size
                )

                if tp_size > 1:
                    model = tensor_parallel_model(model, device_mesh["tp"])
                model = model.to_empty(device="tpu")
                with profiler_range("init weights"):
                    model.init_weights()

                optimizer = create_optimizer(model)
                self.optimizers.append(optimizer)
                self.models.append(model)
                if i == 0:
                    self.model = model
                    self.optimizer = optimizer

    def run(self):
        def loss_fn(output, _target):
            return output.sum()

        if self.interleave == 1:
            stage = PipelineStage(
                self.model,
                self.pp_mesh.get_local_rank(),
                self.pp_size,
                "tpu",
                group=self.pp_mesh.get_group(),
            )

            self.schedule = Schedule1F1B(
                stage, n_microbatches=self.pipeline_length, loss_fn=loss_fn
            )
            self.optimizer.zero_grad(set_to_none=True)
            with (
                torch.autocast("tpu", torch.bfloat16),
                profiler_range("step schedule"),
            ):
                if self.pp_rank == 0:
                    if hasattr(self.model, 'tok_embeddings') or hasattr(self.model, 'module') and hasattr(self.model.module, 'tok_embeddings'):
                        inputs = create_input(self.batch_size, self.sequence_length)
                    else:
                        dim = getattr(self.model, 'cfg', getattr(self.model, 'module', self.model).cfg).dim
                        inputs = torch.randn(self.batch_size, self.sequence_length, dim, device="tpu")
                    self.schedule.step(inputs)
                elif self.pp_rank == self.pp_size - 1:
                    target = torch.zeros([self.batch_size], device="tpu")
                    self.schedule.step(target=target)
                else:
                    self.schedule.step()
            with profiler_range("optimizer"):
                self.optimizer.step()
        else:
            stages = []
            for i, model in enumerate(self.models):
                stages.append(
                    PipelineStage(
                        model,
                        self.pp_mesh.get_local_rank() + i * self.pp_world_size,
                        self.pp_size * self.interleave,
                        "tpu",
                        group=self.pp_mesh.get_group(),
                    )
                )

            self.schedule = ScheduleInterleavedZeroBubble(
                stages, n_microbatches=self.pipeline_length, loss_fn=loss_fn
            )
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            with (
                torch.autocast("tpu", torch.bfloat16),
                profiler_range("step schedule"),
            ):
                if self.pp_rank == 0:
                    if hasattr(self.model, 'tok_embeddings') or hasattr(self.model, 'module') and hasattr(self.model.module, 'tok_embeddings'):
                        inputs = create_input(self.batch_size, self.sequence_length)
                    else:
                        dim = getattr(self.model, 'cfg', getattr(self.model, 'module', self.model).cfg).dim
                        inputs = torch.randn(self.batch_size, self.sequence_length, dim, device="tpu")
                    self.schedule.step(inputs)
                elif self.pp_rank == self.pp_size - 1:
                    target = torch.zeros([self.batch_size], device="tpu")
                    self.schedule.step(target=target)
                else:
                    self.schedule.step()
            with profiler_range("optimizers"):
                for optimizer in self.optimizers:
                    optimizer.step()


@dataclass
class PipelineParallelConfig:
    pp_size: int
    create_model: Callable[[], torch.nn.Module]
    per_gpu_batch_size: int | None
    pipeline_length: int
    tp_size: int
    interleave: int

    def create_model_runner(self, per_gpu_batch_size, sequence_length, compile):
        # disabled torch.compile because
        # 1. it somehow runs slower. pp 4 length 8 runs at 21.6k tokens/gpu/second
        #    without compile and 20.3k tokens/gpu/second with compile
        # 2. it somehow uses more memory, making some configurations unrunnable
        # 3. it takes a long time to compile the model, killing my iteration times
        #
        # We still have to keep it as an argument because in [ModelToRun] this actually
        # does something, and we share the function signature with that one.

        del compile
        return PipelineParallelRunner(
            self.pp_size,
            per_gpu_batch_size,
            sequence_length,
            self.pipeline_length,
            self.create_model,
            self.tp_size,
            self.interleave,
        )

    def with_new_batch_size(self, max_batch_size):
        return PipelineParallelConfig(
            self.pp_size,
            self.create_model,
            max_batch_size,
            self.pipeline_length,
            self.tp_size,
            self.interleave,
        )

    @property
    def batch_per_gpu_divisor(self):
        return (self.pp_size * self.tp_size) / self.pipeline_length

    @property
    def name(self):
        return pipeline_parallel_name(
            self.pp_size, self.pipeline_length, self.tp_size, self.interleave
        )

    @property
    def batch_size(self):
        return self.per_gpu_batch_size

    def batch_sizes_to_run(self):
        # For some reason memory consumption is always higher in the benchmark than in
        # find_max_batch_size. Should probably figure out why, but for now just run with
        # smaller batch sizes.
        smaller_batch_size = max(1, (self.per_gpu_batch_size * 3) // 4)
        return batch_sizes_to_run(smaller_batch_size)


def find_batch_sizes(model_name, models, sequence_length, work_dir, record_memory):
    for model_to_run in models:
        if model_to_run.batch_size is None:
            find_batch_size_for_model(
                model_name,
                model_to_run,
                sequence_length,
                work_dir=work_dir,
                record_memory=record_memory,
            )


def run_tests(
    model_name,
    models,
    arg_batch_size,
    sequence_length,
    work_dir,
    compile,
    pytorch_profiler,
):
    for model_to_run in models:
        if arg_batch_size is not None:
            batch_sizes = arg_batch_size
        elif model_to_run.batch_size is not None:
            batch_sizes = [model_to_run.batch_size]
        else:
            if model_to_run.batch_size is None:
                model_to_run = find_batch_size_for_model(
                    model_name,
                    model_to_run,
                    sequence_length,
                    work_dir=work_dir,
                    record_memory=False,
                )
                if model_to_run.batch_size is None:
                    rank_log(
                        _rank,
                        logger,
                        f"Skipping {model_to_run.name} because it OOMed with batch_size=1.",
                    )
                    continue
                batch_sizes = model_to_run.batch_sizes_to_run()
        for batch_size in batch_sizes:
            with_batch_size = model_to_run.with_new_batch_size(batch_size)
            with profiler_range(
                f"{with_batch_size.name} {with_batch_size.batch_size=}"
            ):
                run_training(
                    with_batch_size,
                    sequence_length,
                    compile=compile,
                    work_dir=work_dir,
                    pytorch_profiler=pytorch_profiler,
                )

    rank_log(_rank, logger, "All training runs successfully completed!")


def measure_model(model_config):
    with torch.device("meta"):
        model = model_config.create_model()
        num_parameters = sum(p.numel() for p in model.parameters())
        parameter_memory = sum(
            find_batch_size.tensor_memory(p) for p in model.parameters()
        )
        num_tensors = sum(1 for _ in model.parameters())
        return num_parameters, parameter_memory, num_tensors


def register_tp_models(model_config, args):
    for tp_size in [2, 4, 8, 16]:  # , 32, 64]:
        if tp_size > _world_size:
            continue
        if _world_size % tp_size != 0:
            # World size needs to be divisible by TP size
            continue

        # create a sharding plan based on the given world_size.
        dp_size = _world_size // tp_size

        # Create a device mesh with 2 dimensions.
        # First dim is the data parallel dimension
        # Second dim is the tensor parallel dimension.
        device_mesh = init_device_mesh(
            "tpu", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )

        tp_mesh = device_mesh["tp"]
        dp_mesh = device_mesh["dp"]

        tp_name = f"tp {tp_size}x"

        def create_tp_model(compile, dp_mesh=dp_mesh, tp_mesh=tp_mesh):
            model = tensor_parallel_model(model_config.create_model(), mesh=tp_mesh)
            if compile:
                model = compile_model(model)
            # turn this off because the sync is broken. It does all the nccl communication at
            # the end so we get unrealistic bad numbers.
            #
            # With this turned off we get unrealistic good numbers, but less wrong. I think we
            # can hide most of the communication if this was done correctly.
            use_dp = False
            if use_dp:
                model = DDP(model, device_mesh=dp_mesh)
            return model

        add_model(
            ModelToRun(
                tp_name,
                create_tp_model,
                dp_mesh.get_rank(),
                None,
                tensor_parallel=tp_size,
            )
        )

        if args.run_fsdp or args.config is not None:
            for ulysses_style in [False, True]:
                fsdp_tp_name = f"fsdp and tp {tp_size}x"
                if ulysses_style:
                    fsdp_tp_name += " ulysses"

                def create_fsdp_tp_model(
                    compile,
                    dp_mesh=dp_mesh,
                    tp_mesh=tp_mesh,
                    ulysses_style=ulysses_style,
                ):
                    if ulysses_style:
                        model = tensor_parallel_model_ulysses(
                            model_config.create_model(), mesh=tp_mesh
                        )
                        model = fully_shard_model(model)
                    else:
                        model = tensor_parallel_model(
                            model_config.create_model(), mesh=tp_mesh
                        )
                        model = fully_shard_model(model, mesh=dp_mesh)
                    if compile:
                        # compiling the whole model doesn't work for some reason. but
                        # compiling each block separately does work and is fast enough.
                        torch.compiler.reset()
                        for i, block in enumerate(model.layers):
                            model.layers[i] = torch.compile(block)
                    return model

                add_model(
                    ModelToRun(
                        fsdp_tp_name,
                        create_fsdp_tp_model,
                        dp_mesh.get_rank(),
                        None,
                        tensor_parallel=tp_size,
                    )
                )


def register_models(mesh_1d, model_config, args):
    if args.run_dp or args.config is not None:

        def create_ddp_model(compile):
            # call to_empty() rightaway because you can't call "DDP()"" on a model that
            # was created on a meta-device (which we sometimes do). This might OOM but
            # that's fine because if we OOM here, we would also OOM later. All the code
            # paths that split the model further (TP or PP) don't go through here.
            model = model_config.create_model().to_empty(device="tpu")
            if compile:
                model = compile_model(model)
            result = DDP(model, device_mesh=mesh_1d)

            def init_weights():
                model.init_weights()

            result.init_weights = init_weights
            return result

        add_model(
            ModelToRun(
                "dp",
                create_ddp_model,
                mesh_1d.get_rank(),
                batch_size=None,
                tensor_parallel=1,
            )
        )
    if args.run_fsdp or args.config is not None:

        def create_fsdp_model(compile):
            model = model_config.create_model().to_empty(device="tpu")
            model = fully_shard_model(model, mesh=mesh_1d)
            if compile:
                model = compile_model(model)
            return model

        add_model(
            ModelToRun(
                "fsdp",
                create_fsdp_model,
                mesh_1d.get_rank(),
                batch_size=None,
                tensor_parallel=1,
            )
        )

    if args.run_pp or args.config is not None:
        for pp_size in [2, 4, 8, 16]:
            for interleaved, tp_size in [(1, 1), (2, 1), (1, 2), (1, 4), (1, 8)]:
                if (pp_size * interleaved) > model_config.n_layers:
                    continue
                if (pp_size * tp_size) > _world_size:
                    continue
                pipeline_lengths = sorted(
                    {pp_size, (pp_size * 3) // 2, 2 * pp_size, max(pp_size, 6)}
                )
                for pipeline_length in pipeline_lengths:
                    add_model(
                        PipelineParallelConfig(
                            pp_size,
                            lambda: model_config.create_model(),
                            None,
                            pipeline_length,
                            tp_size,
                            interleaved,
                        )
                    )

    register_tp_models(model_config, args)


# Written by janeGPT (but I think I saw this on stack overflow before) because type=bool
# behaves weird in Python
def str_to_bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    faulthandler.enable()  # print callstacks on crash
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="+", required=False, default=None)
    parser.add_argument(
        "--batch-size", type=int, nargs="+", required=False, default=None
    )
    parser.add_argument("--sequence-length", type=int, required=False, default=4096)
    parser.add_argument("--model", type=str, default="13b", required=False)
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
    parser.add_argument("--compile", type=str_to_bool, default=True, required=False)
    args = parser.parse_args()
    if args.record_memory and args.work_dir is None:
        raise ValueError("--record-memory requires --work-dir")
    if args.pytorch_profiler and args.work_dir is None:
        raise ValueError("--pytorch-profiler requires --work-dir")

    if False:
        mesh_1d = init_device_mesh("tpu", (_world_size,), mesh_dim_names=("dp",))
        torch.distributed.barrier()
        if _rank == 0:
            print("All started")
    else:
        print(f"calling init_process_group with rank {_rank}")
        torch.distributed.init_process_group(
            backend="tpu_dist",
        )
        print(
            f"done calling init_process_group with rank {_rank}"
        )
        torch.distributed.barrier()
        print(f"done with first barrier on {_rank}")
        if _rank == 0:
            print("All started")
        print(f"calling init_device_mesh with rank {_rank}")
        mesh_1d = init_device_mesh("tpu", (_world_size,), mesh_dim_names=("dp",))

    model_config = get_model_config(args.model)

    num_parameters, parameter_memory, num_tensors = measure_model(model_config)

    rank_log(
        _rank,
        logger,
        f"Running with {num_parameters / 1000**3:.2f}B parameters in {num_tensors} tensors, ({parameter_memory / 1024**3:.2f} GiB)",
    )

    register_models(mesh_1d, model_config, args)

    if args.config is not None:
        models_to_run = [find_model(config) for config in args.config]
    else:
        models_to_run = models
    if args.batch_size is None:
        find_batch_sizes(
            args.model,
            models_to_run,
            sequence_length=args.sequence_length,
            work_dir=args.work_dir,
            record_memory=args.record_memory,
        )
    if args.record_memory:
        # torch.cuda.memory._record_memory_history()
        pass

    rank_log(
        _rank,
        logger,
        f"{csv_prefix}{args.model} compile={args.compile} sequence_length={args.sequence_length}",
    )
    rank_log(
        _rank,
        logger,
        f"{csv_prefix}name,batch_size,peak_memory,batches_per_gpu_per_sec,batches_per_second",
    )
    run_tests(
        args.model,
        models_to_run,
        args.batch_size,
        sequence_length=args.sequence_length,
        work_dir=args.work_dir,
        compile=args.compile,
        pytorch_profiler=args.pytorch_profiler,
    )
    if args.record_memory:
        # torch.cuda.memory._dump_snapshot(f"{args.work_dir}/memory_snapshot.pickle")
        pass

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()