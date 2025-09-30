# Troubleshooting Multi-GPU Kubernetes Training

## Common Issues and Solutions

### 1. Pods Stuck in Pending State

**Symptoms:**
- Pods show `Pending` status
- `kubectl describe pod` shows insufficient GPU resources

**Solutions:**
```bash
# Check GPU allocation
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu

# Check NVIDIA device plugin
kubectl -n kube-system get pods -l app.kubernetes.io/name=nvidia-device-plugin

# Check pod events
kubectl -n ai-lab describe pod <pod-name>