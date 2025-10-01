```bash
#!/bin/bash

echo "🚀 Configurando Jetson Nano para AI Lab..."

# Atualizar sistema
echo "📦 Atualizando pacotes..."
sudo apt update && sudo apt -y upgrade

# Instalar dependências básicas
echo "🔧 Instalando dependências..."
sudo apt -y install python3-pip python3-venv python3-opencv nano tmux htop

# Configurar Python
echo "🐍 Configurando ambiente Python..."
python3 -m pip install --user numpy pycuda

echo "✅ Setup concluído!"