#!/bin/bash
echo "=== Step 3: Deploy NVIDIA Device Plugin with MIG Support ==="

# Add NVIDIA device plugin Helm repository
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

# Install with MIG strategy
helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
    --namespace kube-system \
    --set migStrategy=single \
    --set runtimeClassName=nvidia \
    --create-namespace

# Wait for pods to be ready
echo "Waiting for device plugin pods to be ready..."
kubectl wait --for=condition=ready pod -l app=nvidia-device-plugin-ds -n kube-system --timeout=300s

# Verify GPU resources are visible in Kubernetes
echo "Checking node GPU resources:"
kubectl get nodes -o custom-columns="NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

echo "NVIDIA device plugin deployed successfully."