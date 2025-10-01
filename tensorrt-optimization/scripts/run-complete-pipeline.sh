#!/bin/bash

set -e

echo "🚀 Iniciando Pipeline Completo do Lab 5 - TensorRT Optimization"
echo "================================================================"

# Registrar tempo de início
START_TIME=$(date +%s)

# 1. Setup
echo ""
echo "📍 Passo 1/6: Configurando ambiente..."
./scripts/setup-environment.sh

# 2. Export ONNX
echo ""
echo "📍 Passo 2/6: Exportando modelo ONNX..."
./scripts/export-onnx.sh

# 3. Build TensorRT Engines
echo ""
echo "📍 Passo 3/6: Construindo engines TensorRT..."
./scripts/build-tensorrt-engines.sh

# 4. Start Triton Server
echo ""
echo "📍 Passo 4/6: Iniciando Triton Server..."
./scripts/start-triton-server.sh

# 5. Run Benchmarks
echo ""
echo "📍 Passo 5/6: Executando benchmarks..."
./scripts/run-benchmarks.sh

# 6. Validation
echo ""
echo "📍 Passo 6/6: Validando correção..."
./scripts/validate-correctness.sh

# Tempo total
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "================================================================"
echo "✅ Pipeline Completo Concluído!"
echo "⏱️  Tempo total: $((DURATION / 60)) minutos e $((DURATION % 60)) segundos"
echo ""
echo "📊 Resultados disponíveis em:"
echo "   - Benchmarks: ../benchmarks/latest/"
echo "   - Modelos: ../models/"
echo "   - Logs: docker logs triton-server"
echo ""
echo "🔧 Comandos úteis:"
echo "   ./scripts/cleanup.sh          # Limpar ambiente"
echo "   docker logs -f triton-server  # Ver logs do servidor"
echo "   ./scripts/run-benchmarks.sh   # Re-executar benchmarks"