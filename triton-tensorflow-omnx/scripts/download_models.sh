#!/bin/bash
echo "=== Baixando Modelos ==="

# Criar diretórios
mkdir -p models/tf_model/1
mkdir -p models/onnx_model/1

# Baixar modelo ONNX
echo "Baixando MobileNetV2 ONNX..."
wget -O models/onnx_model/1/mobilenetv2-7.onnx \
    https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx

echo "✅ Modelos baixados!"
echo "⚠️  NOTA: Você precisa adicionar manualmente o modelo TensorFlow SavedModel em models/tf_model/1/"