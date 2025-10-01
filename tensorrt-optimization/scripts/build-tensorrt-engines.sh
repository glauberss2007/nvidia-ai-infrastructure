#!/bin/bash

set -e

echo "🔨 Construindo engines TensorRT..."

# Verificar se o arquivo ONNX existe
if [ ! -f "../models/resnet50.onnx" ]; then
    echo "❌ Arquivo ONNX não encontrado. Execute primeiro: ./scripts/export-onnx.sh"
    exit 1
fi

# Criar diretório de modelos se não existir
mkdir -p ../models

# Construir engine FP16
echo "🔄 Construindo engine FP16..."
docker run --rm --gpus all \
    -v $(pwd)/..:/workspace \
    -w /workspace \
    nvcr.io/nvidia/tritonserver:24.03-py3 \
    trtexec --onnx=models/resnet50.onnx --saveEngine=models/resnet50_fp16.plan \
    --fp16 --explicitBatch \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:8x3x224x224 \
    --maxShapes=input:32x3x224x224 \
    --workspace=4096

echo "✅ Engine FP16 construída: ../models/resnet50_fp16.plan"

# Verificar se existem imagens de calibração para INT8
CALIB_COUNT=$(find ../calibration -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)

if [ "$CALIB_COUNT" -ge 10 ]; then
    echo "🔄 Construindo engine INT8 com $CALIB_COUNT imagens de calibração..."
    
    # Instalar dependências Python adicionais se necessário
    docker run --rm --gpus all \
        -v $(pwd)/..:/workspace \
        -w /workspace \
        nvcr.io/nvidia/tritonserver:24.03-py3 \
        python src/build_int8_engine.py
else
    echo "⚠️  Pulando construção INT8 - imagens de calibração insuficientes ($CALIB_COUNT)"
    echo "   Adicione imagens em ../calibration/ e execute novamente"
fi

echo "✅ Construção de engines concluída!"
ls -la ../models/