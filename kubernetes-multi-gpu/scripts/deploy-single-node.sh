#!/bin/bash

set -e

echo "🚀 Deploying Single-Node Multi-GPU Training Job..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Apply ConfigMap
echo -e "${BLUE}📦 Creating training script ConfigMap...${NC}"
kubectl apply -f ../manifests/02-configmap.yaml

# Apply Single-node Job
echo -e "${BLUE}🎯 Deploying single-node job...${NC}"
kubectl apply -f ../manifests/03-single-node-job.yaml

echo -e "${GREEN}✅ Single-node job deployed!${NC}"
echo ""
echo -e "${YELLOW}📊 Monitoring commands:${NC}"
echo -e "   Watch logs:    kubectl -n ai-lab logs -f job/ddp-1node"
echo -e "   Check status:  kubectl -n ai-lab get pods -l job-name=ddp-1node"
echo -e "   GPU usage:     kubectl -n ai-lab exec <pod-name> -- nvidia-smi"
echo ""
echo -e "${YELLOW}🧹 Cleanup command:${NC}"
echo -e "   kubectl -n ai-lab delete job/ddp-1node"