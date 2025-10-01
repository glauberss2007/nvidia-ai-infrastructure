#!/bin/bash

set -e

echo "📊 Executando benchmarks de performance..."

# Verificar se o servidor está rodando
if ! docker ps | grep -q triton-server; then
    echo "❌ Triton Server não está rodando. Execute primeiro: ./scripts/start-triton-server.sh"
    exit 1
fi

# Criar diretório para resultados
mkdir -p ../benchmarks
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="../benchmarks/results_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "📁 Salvando resultados em: $RESULTS_DIR"

# Função para executar benchmark
run_benchmark() {
    local model_name=$1
    local batch_size=$2
    local concurrency=$3
    local output_file="$RESULTS_DIR/${model_name}_b${batch_size}_c${concurrency}.txt"
    
    echo "🧪 Benchmark: $model_name, batch=$batch_size, concurrency=$concurrency"
    
    docker exec -it triton-server \
        perf_analyzer -m "$model_name" -b "$batch_size" -u localhost:8001 -i grpc \
        --concurrency-range "$concurrency" --measurement-mode count_windows \
        --output-shared-memory-size 1000000 > "$output_file" 2>&1
        
    # Extrair métricas principais
    latency=$(grep "p50 latency" "$output_file" | awk '{print $3}' || echo "N/A")
    throughput=$(grep "Inferences/Second" "$output_file" | awk '{print $3}' || echo "N/A")
    
    echo "   📈 Throughput: $throughput inf/s, Latency p50: $latency ms"
}

# Benchmark para cada modelo disponível
MODELS=("resnet50_fp32" "resnet50_trt_fp16")
if [ -f "../model_repository/resnet50_trt_int8/config.pbtxt" ]; then
    MODELS+=("resnet50_trt_int8")
fi

echo "🔍 Modelos detectados: ${MODELS[*]}"

# Executar suite de benchmarks
for model in "${MODELS[@]}"; do
    echo ""
    echo "🚀 Benchmarking $model..."
    
    # Diferentes configurações de batch e concorrência
    run_benchmark "$model" 1 1:32:4
    run_benchmark "$model" 8 1:16:2
    run_benchmark "$model" 16 1:8:2
    run_benchmark "$model" 32 1:4:1
    
    sleep 2  # Pequena pausa entre modelos
done

# Gerar relatório consolidado
echo "📋 Gerando relatório consolidado..."
cat > "$RESULTS_DIR/summary.md" << 'EOF'
# Relatório de Performance - TensorRT Optimization

## Configuração
- **GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
- **Data**: $(date)
- **Modelos testados**: ${MODELS[*]}

## Resultados

| Modelo | Batch Size | Concurrency | Throughput (inf/s) | Latency p50 (ms) |
|--------|------------|-------------|-------------------|------------------|
EOF

for result_file in "$RESULTS_DIR"/*.txt; do
    if [ -f "$result_file" ]; then
        model=$(basename "$result_file" | cut -d'_' -f1-3)
        batch=$(echo "$result_file" | grep -o 'b[0-9]*' | sed 's/b//')
        conc=$(echo "$result_file" | grep -o 'c[0-9]*' | sed 's/c//')
        throughput=$(grep "Inferences/Second" "$result_file" | awk '{print $3}' || echo "N/A")
        latency=$(grep "p50 latency" "$result_file" | awk '{print $3}' || echo "N/A")
        
        echo "| $model | $batch | $conc | $throughput | $latency |" >> "$RESULTS_DIR/summary.md"
    fi
done

echo ""
echo "✅ Benchmarks concluídos!"
echo "📊 Relatório: $RESULTS_DIR/summary.md"
echo ""
echo "📈 Para visualização:"
echo "   cat $RESULTS_DIR/summary.md"