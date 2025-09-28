import torch
import numpy as np

def check_gpu():
    """Verifica se a GPU está disponível"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ {gpu_count} GPU(s) disponível(is):")
        for i in range(gpu_count):
            print(f"   {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("❌ GPU não disponível - usando CPU")
        return False

def test_model():
    """Testa se o modelo pode ser carregado"""
    try:
        # Verificar se o modelo existe
        import os
        if os.path.exists('model_repository/cifar_model/1/model.pt'):
            print("✅ Modelo encontrado!")
            return True
        else:
            print("❌ Modelo não encontrado. Execute train.py primeiro.")
            return False
    except Exception as e:
        print(f"❌ Erro ao verificar modelo: {e}")
        return False

if __name__ == "__main__":
    check_gpu()
    test_model()