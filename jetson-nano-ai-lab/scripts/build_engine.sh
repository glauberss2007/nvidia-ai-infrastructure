#!/bin/bash

echo "🔨 Building TensorRT engine..."

# Verificar se trtexec está disponível
if ! command -v trtexec &> /dev/null; then
    echo "❌ trtexec não encontrado. Verifique a instalação do TensorRT."
    exit 1
fi

# Build do engine FP16
trtexec \
    --onnx=resnet50.onnx \
    --saveEngine=resnet50_fp16.plan \
    --fp16 \
    --explicitBatch \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:8x3x224x224 \
    --maxShapes=input:32x3x224x224 \
    --workspace=2048

echo "✅ Engine built: resnet50_fp16.plan"