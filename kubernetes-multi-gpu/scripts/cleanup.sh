#!/bin/bash

echo "🧹 Cleaning up Multi-GPU Training resources..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}⚠️  This will delete all resources in the ai-lab namespace${NC}"
read -p "Are you sure? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 1
fi

echo -e "${YELLOW}🗑️  Deleting resources...${NC}"

# Delete single-node job
kubectl -n ai-lab delete job/ddp-1node --ignore-not-found=true

# Delete multi-node resources
kubectl -n ai-lab delete statefulset/ddp --ignore-not-found=true
kubectl -n ai-lab delete service/ddp-hs --ignore-not-found=true

# Delete configmap and PVC
kubectl -n ai-lab delete configmap/ddp-train --ignore-not-found=true
kubectl -n ai-lab delete pvc/ddp-outputs --ignore-not-found=true

# Delete namespace
kubectl delete namespace ai-lab --ignore-not-found=true

echo -e "${GREEN}✅ Cleanup completed!${NC}"
echo ""
echo -e "${YELLOW}📋 Verification:${NC}"
echo "   kubectl get namespaces | grep ai-lab"
echo "   kubectl get all -n ai-lab"