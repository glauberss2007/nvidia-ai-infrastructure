import torch
import os
import sys

def convert_model_for_triton(model, model_path):
    """Converte o modelo PyTorch para formato Triton"""
    
    # Cria diretório do modelo Triton
    model_repo_path = "src/inference/model_repository/my_model/1"
    os.makedirs(model_repo_path, exist_ok=True)
    
    # Salva o modelo no formato esperado pelo Triton
    triton_model_path = os.path.join(model_repo_path, "model.pt")
    
    # Cria um exemplo de input para tracing
    example_input = torch.randn(1, 3, 32, 32)
    
    # Usa tracing para criar um modelo compatível com TorchScript
    traced_model = torch.jit.trace(model, example_input)
    
    # Salva o modelo traced
    torch.jit.save(traced_model, triton_model_path)
    
    print(f"✅ Model converted and saved to: {triton_model_path}")
    print(f"   Model input shape: {example_input.shape}")
    
    return triton_model_path