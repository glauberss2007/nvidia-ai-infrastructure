#!/bin/bash

echo "🚀 Configurando ambiente para AI Pipeline Lab..."

# Verificar se GPU está disponível
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU NVIDIA detectada"
    nvidia-smi
else
    echo "⚠️  GPU NVIDIA não detectada - usando CPU"
fi

# Criar ambiente virtual
echo "📦 Criando ambiente virtual..."
python -m venv venv
source venv/bin/activate

# Instalar dependências
echo "📚 Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

# Baixar dataset
echo "📥 Baixando dataset CIFAR-10..."
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='./data', download=True)"

# Criar diretórios necessários
mkdir -p model_repository/my_model/1
mkdir -p logs

echo "✅ Configuração concluída!"
echo "📖 Próximos passos:"
echo "   1. source venv/bin/activate"
echo "   2. python src/train_model.py"
echo "   3. docker-compose up triton-server"
echo "   4. python src/inference_client.py"