```bash
#!/bin/bash
echo "=== Configurando Ambiente para Lab Triton ==="

# Verificar GPU
echo "1. Verificando GPU..."
nvidia-smi || { echo "GPU não detectada!"; exit 1; }

# Verificar Docker
echo "2. Verificando Docker..."
docker --version || { echo "Docker não instalado!"; exit 1; }

# Baixar container do Triton
echo "3. Baixando container Triton..."
docker pull nvcr.io/nvidia/tritonserver:23.12-py3

echo "✅ Setup completo!"