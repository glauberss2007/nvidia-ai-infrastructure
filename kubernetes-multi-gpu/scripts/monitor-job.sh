#!/bin/bash

echo "📊 Monitoring Multi-GPU Training Jobs..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔍 Checking cluster status...${NC}"
echo ""

# Check nodes and GPUs
echo -e "${YELLOW}🏠 NODES & GPU ALLOCATION:${NC}"
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory

echo ""
echo -e "${YELLOW}📦 RUNNING PODS:${NC}"
kubectl -n ai-lab get pods -o wide

echo ""
echo -e "${YELLOW}🎯 JOBS:${NC}"
kubectl -n ai-lab get jobs

echo ""
echo -e "${YELLOW}🏗️ STATEFULSETS:${NC}"
kubectl -n ai-lab get statefulsets

echo ""
echo -e "${YELLOW}🔧 SERVICES:${NC}"
kubectl -n ai-lab get services

echo ""
echo -e "${YELLOW}💾 CONFIGMAPS:${NC}"
kubectl -n ai-lab get configmaps

echo ""
echo -e "${YELLOW}📈 NVIDIA DEVICE PLUGIN:${NC}"
kubectl -n kube-system get pods -l app.kubernetes.io/name=nvidia-device-plugin

echo ""
echo -e "${GREEN}🚀 Quick monitoring commands:${NC}"
echo "   Single-node logs: kubectl -n ai-lab logs -f job/ddp-1node"
echo "   Multi-node logs:  kubectl -n ai-lab logs -f ddp-0"
echo "   GPU usage:        kubectl -n ai-lab exec <pod-name> -- nvidia-smi"
echo "   Pod details:      kubectl -n ai-lab describe pod <pod-name>"