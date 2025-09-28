import numpy as np
import tritonclient.http as httpclient
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

class TritonClient:
    """Cliente para inference com Triton Server"""
    
    def __init__(self, url="localhost:8000"):
        self.client = httpclient.InferenceServerClient(url=url)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def predict(self, image):
        """Faz predição em uma imagem"""
        # Pré-processamento
        if isinstance(image, str):
            # Se for path de arquivo
            image = Image.open(image)
        elif isinstance(image, bytes):
            # Se for bytes
            image = Image.open(BytesIO(image))
        
        image = self.transform(image).unsqueeze(0).numpy()
        
        # Preparar inputs para Triton
        inputs = httpclient.InferInput(
            "INPUT__0", 
            image.shape, 
            "FP32"
        )
        inputs.set_data_from_numpy(image.astype(np.float32))
        
        # Preparar outputs
        outputs = httpclient.InferRequestedOutput("OUTPUT__0")
        
        # Fazer inference
        try:
            result = self.client.infer(
                model_name="my_model",
                inputs=[inputs],
                outputs=[outputs]
            )
            
            output = result.as_numpy("OUTPUT__0")
            predicted_class = np.argmax(output, axis=1)
            
            return {
                "predicted_class": predicted_class[0],
                "probabilities": output[0],
                "confidence": np.max(output[0])
            }
            
        except Exception as e:
            print(f"Erro na inference: {e}")
            return None

def test_inference():
    """Testa o cliente de inference"""
    client = TritonClient("localhost:8000")
    
    # Testar com imagem aleatória (simulação)
    random_image = np.random.rand(1, 3, 32, 32).astype(np.float32)
    
    # Criar input manual para teste
    inputs = httpclient.InferInput("INPUT__0", random_image.shape, "FP32")
    inputs.set_data_from_numpy(random_image)
    
    outputs = httpclient.InferRequestedOutput("OUTPUT__0")
    
    try:
        result = client.client.infer("my_model", inputs=[inputs], outputs=[outputs])
        output = result.as_numpy("OUTPUT__0")
        print(f"Predição: {np.argmax(output)}")
        print(f"Probabilidades: {output}")
    except Exception as e:
        print(f"Servidor não disponível: {e}")

if __name__ == "__main__":
    test_inference()