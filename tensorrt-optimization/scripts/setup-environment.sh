#!/bin/bash

set -e

echo "🚀 Configurando ambiente para Lab 5 - TensorRT Optimization..."

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Verificar Docker
echo -e "${BLUE}🔍 Verificando Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}❌ Docker não encontrado. Por favor instale Docker primeiro.${NC}"
    exit 1
fi

# Verificar NVIDIA Container Toolkit
echo -e "${BLUE}🔍 Verificando NVIDIA Container Toolkit...${NC}"
if ! docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}❌ NVIDIA Container Toolkit não configurado corretamente.${NC}"
    exit 1
fi

# Criar diretórios
echo -e "${BLUE}📁 Criando estrutura de diretórios...${NC}"
mkdir -p ../models
mkdir -p ../calibration
mkdir -p ../benchmarks
mkdir -p ../model_repository

# Pull de containers necessários
echo -e "${BLUE}📦 Baixando containers NVIDIA...${NC}"
docker pull nvcr.io/nvidia/pytorch:24.03-py3
docker pull nvcr.io/nvidia/tritonserver:24.03-py3

echo -e "${GREEN}✅ Ambiente configurado com sucesso!${NC}"
echo ""
echo -e "${YELLOW}📝 Próximos passos:${NC}"
echo -e "   ./scripts/export-onnx.sh    # Exportar modelo para ONNX"
echo -e "   ./scripts/build-tensorrt-engines.sh # Construir engines TensorRT"