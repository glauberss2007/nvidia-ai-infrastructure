#!/bin/bash

set -e

echo "🚀 Starting MIG setup for NVIDIA A100..."

# Step 1: Enable MIG
echo "✅ Step 1: Enabling MIG mode on GPU 0..."
sudo nvidia-smi -i 0 -mig 1
echo "⚠️  Reboot may be required if MIG was not previously enabled."

# Step 2: Create MIG instances
echo "✅ Step 2: Creating 3x MIG instances of type 2g.10gb..."
sudo nvidia-smi mig -cgi 0,1,2 -gi 1 -i 0

# Step 3: Verify MIG instances
echo "✅ Step 3: Verifying MIG instances..."
echo "📋 Available MIG profiles:"
nvidia-smi mig -lgip

echo "📋 Created MIG instances:"
nvidia-smi -L | grep MIG

# Step 4: Install NVIDIA Device Plugin via Helm
echo "✅ Step 4: Installing NVIDIA Device Plugin with MIG support..."
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --set migStrategy=single \
  --set runtimeClassName=nvidia \
  --namespace nvidia-device-plugin \
  --create-namespace

# Step 5: Verify GPU resources in Kubernetes
echo "✅ Step 5: Waiting for device plugin to be ready..."
sleep 30
echo "📋 GPU resources available in Kubernetes:"
kubectl get nodes -o custom-columns="NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

echo "🎉 MIG setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Deploy a test pod: kubectl apply -f kubernetes/mig-pod.yaml"
echo "2. Verify pod placement: kubectl describe pod mig-test"
echo "3. Check MIG usage: nvidia-smi"