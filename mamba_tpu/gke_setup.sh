#!/bin/bash
# gke_setup.sh - Script to create a GKE cluster and node pool for TPU Ironwood (v5litepod-16)

set -e

PROJECT_ID="cloud-tpu-multipod-dev"
REGION="us-central1"
ZONE="us-central1"
CLUSTER_NAME="rick-tpu-ctk"
NODEPOOL_NAME="ironwood-2x2x4-pool"

# echo "Creating GKE cluster ${CLUSTER_NAME} in ${ZONE}..."
# gcloud container clusters create ${CLUSTER_NAME} \
#   --location=${ZONE} \
#   --release-channel="regular" \
#   --workload-pool="${PROJECT_ID}.svc.id.goog" \
#   --enable-ip-alias

POLICY_NAME="${NODEPOOL_NAME}-policy"
echo "Creating compute resource policy ${POLICY_NAME} in ${REGION}..."
gcloud compute resource-policies create workload-policy ${POLICY_NAME} \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=2x2x4 \
  --project=${PROJECT_ID} \
  --region=${REGION} || echo "Policy might already exist, continuing..."

echo "Creating TPU v7 ironwood nodepool ${NODEPOOL_NAME}..."
# Ironwood 2x2x4 shape corresponds to 16 chips, which requires 4 VMs with 4 TPUs each
gcloud container node-pools create ${NODEPOOL_NAME} \
  --cluster=${CLUSTER_NAME} \
  --location=${ZONE} \
  --node-locations=us-central1-c \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=4 \
  --tpu-topology=2x2x4 \
  --spot \
  --placement-policy=${POLICY_NAME} \
  --enable-image-streaming \
  --scopes=https://www.googleapis.com/auth/cloud-platform

echo "Cluster and nodepool created successfully."
echo "Ensure you have the JobSet CRD installed on your cluster if you haven't already:"
echo "kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.6.0/manifests.yaml"
