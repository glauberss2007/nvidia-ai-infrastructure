#!/bin/bash

echo "🧹 Limpando ambiente..."

# Parar e remover container Triton
if docker ps | grep -q triton-server; then
    echo "🛑 Parando Triton Server..."
    docker stop triton-server
    docker rm triton-server
fi

# Remover arquivos temporários (opcional)
read -p "❓ Deseja remover modelos e resultados? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removendo arquivos gerados..."
    rm -rf ../models/*
    rm -rf ../model_repository/*
    rm -rf ../benchmarks/latest
    echo "✅ Arquivos removidos"
else
    echo "📁 Arquivos preservados em ../models/ e ../benchmarks/"
fi

echo "✨ Limpeza concluída!"