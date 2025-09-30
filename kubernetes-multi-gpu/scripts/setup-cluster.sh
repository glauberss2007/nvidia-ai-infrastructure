#!/bin/bash

set -e

echo "🚀 Setting up Kubernetes cluster for Multi-GPU Training..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check command existence
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 not found. Please install it first.${NC}"
        exit 1
    fi
}

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
check_command kubectl
check_command helm

# Verify cluster connection
echo -e "${BLUE}🔗 Checking Kubernetes cluster connection...${NC}"
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Connected to Kubernetes cluster${NC}"

# Create namespace
echo -e "${BLUE}📁 Creating namespace...${NC}"
kubectl apply -f ../manifests/00-namespace.yaml

# Install NVIDIA Device Plugin if not exists
echo -e "${BLUE}📦 Installing NVIDIA Device Plugin...${NC}"
if ! kubectl get daemonset nvidia-device-plugin -n kube-system &> /dev/null; then
    helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
    helm repo update
    helm upgrade --install nvidia-device-plugin nvdp/nvidia-device-plugin \
        --namespace kube-system \
        --set mig.strategy=single
    echo -e "${GREEN}✅ NVIDIA Device Plugin installed${NC}"
else
    echo -e "${YELLOW}⚠️  NVIDIA Device Plugin already installed${NC}"
fi

# Wait for device plugin to be ready
echo -e "${BLUE}⏳ Waiting for NVIDIA Device Plugin to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=nvidia-device-plugin -n kube-system --timeout=120s

# Verify GPU nodes
echo -e "${BLUE}🔍 Checking GPU nodes...${NC}"
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu

echo -e "${GREEN}✅ Cluster setup completed!${NC}"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo -e "   Single-node: ./scripts/deploy-single-node.sh"
echo -e "   Multi-node:  ./scripts/deploy-multi-node.sh"