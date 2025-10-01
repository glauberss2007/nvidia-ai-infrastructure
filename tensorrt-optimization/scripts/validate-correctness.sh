#!/bin/bash

set -e

echo "🔍 Validando correção entre modelos..."

# Verificar se o servidor está rodando
if ! docker ps | grep -q triton-server; then
    echo "❌ Triton Server não está rodando. Execute primeiro: ./scripts/start-triton-server.sh"
    exit 1
fi

# Verificar se existem imagens para teste
CALIB_COUNT=$(find ../calibration -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)

if [ "$CALIB_COUNT" -lt 1 ]; then
    echo "⚠️  Baixando imagens de exemplo para validação..."
    
    # Criar diretório de calibração
    mkdir -p ../calibration
    
    # Download de algumas imagens de exemplo (opcional)
    echo "📥 Para melhores resultados, adicione suas próprias imagens em ../calibration/"
    echo "   Ou execute: wget/png de imagens JPEG/PNG para o diretório calibration"
fi

# Executar validação
echo "🧪 Executando validação de correção..."
docker run --rm --network host \
    -v $(pwd)/..:/workspace \
    -w /workspace \
    nvcr.io/nvidia/tritonserver:24.03-py3 \
    python src/validation_client.py --num-images 3

echo "✅ Validação concluída!"