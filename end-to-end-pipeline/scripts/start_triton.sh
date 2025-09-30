#!/bin/bash

echo "🚀 Starting Triton Inference Server..."

# Verifica se Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não encontrado. Por favor, instale Docker primeiro."
    exit 1
fi

# Verifica se NVIDIA Container Toolkit está instalado
if ! docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Container Toolkit não configurado corretamente."
    echo "   Por favor, instale: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    exit 1
fi

# Verifica se o repositório de modelos existe
if [ ! -d "src/inference/model_repository" ]; then
    echo "❌ Model repository não encontrado. Execute o pipeline primeiro:"
    echo "   python run_pipeline.py"
    exit 1
fi

echo "📁 Model repository: src/inference/model_repository"
echo "🌐 Triton endpoints:"
echo "   - HTTP: localhost:8080"
echo "   - gRPC: localhost:8081"
echo "   - Metrics: localhost:8082"

# Inicia o servidor Triton
docker run --gpus all --rm -p 8080:8080 -p 8081:8081 -p 8082:8082 \
    -v $(pwd)/src/inference/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.03-py3 \
    tritonserver --model-repository=/models --log-verbose=1