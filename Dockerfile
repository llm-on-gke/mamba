# Use the Torch TPU base image
FROM us-east1-docker.pkg.dev/cloud-tpu-multipod-dev/gke-llm/torchtpu-torchtitan

# Install system dependencies
RUN apt-get update && apt-get install -y curl gnupg nano git build-essential

WORKDIR /workspace

# Install Mamba dependencies
RUN pip install  einops transformers

# Copy the entire mono-repo
COPY . /workspace/repo

# Copy mamba_tpu and mamba_ssm out of the repo for execution
RUN cp -r /workspace/repo/mamba_tpu /workspace/mamba_tpu
RUN cp -r /workspace/repo/mamba_ssm /workspace/mamba_ssm

# Clean up repo source to save space
RUN rm -rf /workspace/repo

ENV PYTHONPATH="/workspace:/workspace/mamba_ssm:/workspace/mamba_tpu"
ENV PYTHONUNBUFFERED=1

CMD ["python3", "mamba_tpu/model_parrallel_playground.py"]
