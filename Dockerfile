# Use a Python base image
FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y curl gnupg nano git build-essential

# Ensure python3 is available in /usr/bin for Bazel shims and local scripts.
RUN ln -sf $(which python) /usr/bin/python3

# Install bazelisk for bazel management
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

WORKDIR /workspace

# Install PyTorch (CPU version)
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# Install torch_tpu dependencies
RUN pip install \
    numpy \
    scipy \
    absl-py \
    "jax>=0.9.1" \
    libtpu \
    portpicker \
    transformers \
    "etils[epath]" \
    expecttest \
    tensorboard \
    frozendict \
    flax \
    pillow

# Install Mamba dependencies
RUN pip install einops

# Copy the entire mono-repo
COPY . /workspace/repo

# Build the torch_tpu wheel
RUN cd /workspace/repo/torch_tpu && \
    rm -rf bazel-* && \
    bazel build -c opt //ci/wheel:torch_tpu_wheel --config=no_rbe --jobs=8 --local_ram_resources=16384 && \
    mkdir -p dist && mv bazel-bin/ci/wheel/* dist/ && \
    pip install dist/*.whl --no-deps

# Copy mamba_tpu and mamba_ssm out of the repo for execution
RUN cp -r /workspace/repo/mamba_tpu /workspace/mamba_tpu
RUN cp -r /workspace/repo/mamba_ssm /workspace/mamba_ssm

# Clean up repo source to save space
RUN rm -rf /workspace/repo

ENV PYTHONPATH="/workspace:/workspace/mamba_ssm:/workspace/mamba_tpu"
ENV PYTHONUNBUFFERED=1

CMD ["python3", "mamba_tpu/model_parrallel_playground.py"]
