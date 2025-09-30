#!/bin/bash

echo "🚀 Setting up End-to-End AI Pipeline Environment..."

# Verifica se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado. Por favor, instale Python 3.8+"
    exit 1
fi

# Verifica se pip está instalado
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 não encontrado. Por favor, instale pip3"
    exit 1
fi

# Cria virtual environment (opcional)
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Instala dependências
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Cria diretórios necessários
echo "📁 Creating directories..."
mkdir -p data models logs

# Verifica GPU
echo "🔍 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "⚠️  NVIDIA GPU not detected. Running in CPU mode."
fi

echo "✅ Environment setup completed!"
echo "   To activate virtual environment: source venv/bin/activate"
echo "   To run pipeline: python run_pipeline.py"