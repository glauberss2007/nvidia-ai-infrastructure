#!/bin/bash

set -e

echo "🚀 Exportando modelo PyTorch para ONNX..."

# Executar script de exportação no container
docker run --rm --gpus all \
    -v $(pwd)/..:/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:24.03-py3 \
    python src/export_onnx.py

echo "✅ Exportação ONNX concluída!"
echo "📁 Arquivo: ../models/resnet50.onnx"