#!/usr/bin/env python3
"""
Script para exportar modelo ResNet50 para ONNX com suporte a batch dinâmico
"""

import torch
import torchvision.models as models
import os

def main():
    print("🚀 Exportando ResNet50 para ONNX...")
    
    # Criar diretório de modelos se não existir
    os.makedirs("../models", exist_ok=True)
    
    # Carregar modelo pré-treinado
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = model.eval().cuda()
    
    # Input dummy para tracing
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    
    # Exportar para ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        "../models/resnet50.onnx",
        input_names=["input"], 
        output_names=["logits"],
        opset_version=13, 
        do_constant_folding=True,
        dynamic_axes={
            'input': {0: 'batch'},
            'logits': {0: 'batch'}
        }
    )
    
    print("✅ Modelo ONNX exportado: ../models/resnet50.onnx")
    
    # Verificar tamanho do arquivo
    file_size = os.path.getsize("../models/resnet50.onnx") / (1024 * 1024)
    print(f"📊 Tamanho do arquivo: {file_size:.2f} MB")

if __name__ == "__main__":
    main()