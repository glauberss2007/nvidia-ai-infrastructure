#!/bin/bash

echo "🚀 Running End-to-End AI Pipeline..."

# Ativa virtual environment se existir
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Executa o pipeline
python run_pipeline.py

# Verifica se foi bem sucedido
if [ $? -eq 0 ]; then
    echo "✅ Pipeline executed successfully!"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Start Triton server: ./scripts/start_triton.sh"
    echo "   2. Test inference: python -c \"from src.inference.triton_client import test_triton_connection; test_triton_connection()\""
    echo "   3. Monitor GPU: python -c \"from src.utils.monitoring import monitor_gpu; monitor_gpu(30)\""
else
    echo "❌ Pipeline execution failed!"
    exit 1
fi