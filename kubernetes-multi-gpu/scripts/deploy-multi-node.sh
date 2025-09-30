#!/bin/bash

set -e

echo "🚀 Deploying Multi-Node Multi-GPU Training..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Apply ConfigMap
echo -e "${BLUE}📦 Creating training script ConfigMap...${NC}"
kubectl apply -f ../manifests/02-configmap.yaml

# Apply Headless Service
echo -e "${BLUE}🔗 Creating headless service...${NC}"
kubectl apply -f ../manifests/04-multi-node/service.yaml

# Apply StatefulSet
echo -e "${BLUE}🏗️ Deploying StatefulSet...${NC}"
kubectl apply -f ../manifests/04-multi-node/statefulset.yaml

echo -e "${GREEN}✅ Multi-node deployment completed!${NC}"
echo ""
echo -e "${YELLOW}📊 Monitoring commands:${NC}"
echo -e "   Check pods:    kubectl -n ai-lab get pods -l app=ddp -o wide"
echo -e "   Pod 0 logs:    kubectl -n ai-lab logs -f ddp-0"
echo -e "   Pod 1 logs:    kubectl -n ai-lab logs -f ddp-1"
echo -e "   GPU usage:     kubectl -n ai-lab exec ddp-0 -- nvidia-smi"
echo ""
echo -e "${YELLOW}🔧 Debugging commands:${NC}"
echo -e "   Describe pod:  kubectl -n ai-lab describe pod ddp-0"
echo -e "   Check events:  kubectl -n ai-lab get events --sort-by=.lastTimestamp"