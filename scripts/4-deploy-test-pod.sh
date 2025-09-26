#!/bin/bash
echo "=== Step 4: Deploy Test Pod to Verify MIG ==="

# Deploy the test pod
kubectl apply -f ../manifests/mig-pod.yaml

# Wait for pod to be running
echo "Waiting for pod to start..."
kubectl wait --for=condition=ready pod/mig-test --timeout=60s

# Verify pod placement and GPU allocation
echo "Pod details:"
kubectl describe pod mig-test

echo "Checking GPU usage on node:"
kubectl exec -it mig-test -- nvidia-smi

echo "Test pod deployed successfully."