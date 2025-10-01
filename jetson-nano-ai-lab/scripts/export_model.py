#!/usr/bin/env python3

import torch
import torchvision as tv

def export_resnet50():
    print("🔄 Exportando ResNet50 para ONNX...")
    
    # Carregar modelo pré-treinado
    model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT)
    model = model.eval().cuda()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    
    # Exportar para ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "resnet50.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"}
        }
    )
    
    print("✅ Modelo exportado: resnet50.onnx")

if __name__ == "__main__":
    export_resnet50()