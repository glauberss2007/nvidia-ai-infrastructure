import numpy as np
import tritonclient.http as httpclient
import torch
from torchvision import transforms
from PIL import Image
import requests

class TritonInference:
    def __init__(self, url="localhost:8000"):
        self.client = httpclient.InferenceServerClient(url=url)
        self.labels = ['avião', 'carro', 'pássaro', 'gato', 'veado', 
                      'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']
    
    def predict_image(self, image_path):
        """Faz predição em uma imagem"""
        try:
            # Carregar e pré-processar imagem
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image_tensor = transform(image).unsqueeze(0).numpy()
            
            # Preparar input para Triton
            inputs = httpclient.InferInput(
                "INPUT__0", 
                image_tensor.shape, 
                "FP32"
            )
            inputs.set_data_from_numpy(image_tensor.astype(np.float32))
            
            # Preparar output
            outputs = httpclient.InferRequestedOutput("OUTPUT__0")
            
            # Fazer inference
            result = self.client.infer(
                model_name="cifar_model",
                inputs=[inputs],
                outputs=[outputs]
            )
            
            # Processar resultado
            output = result.as_numpy("OUTPUT__0")
            predicted_class = np.argmax(output, axis=1)[0]
            confidence = np.max(output[0])
            
            return {
                "class": self.labels[predicted_class],
                "class_id": predicted_class,
                "confidence": float(confidence),
                "all_probabilities": output[0].tolist()
            }
            
        except Exception as e:
            print(f"❌ Erro na inference: {e}")
            return None
    
    def test_random(self):
        """Testa com imagem aleatória"""
        random_image = np.random.rand(1, 3, 32, 32).astype(np.float32)
        
        inputs = httpclient.InferInput("INPUT__0", random_image.shape, "FP32")
        inputs.set_data_from_numpy(random_image)
        
        outputs = httpclient.InferRequestedOutput("OUTPUT__0")
        
        try:
            result = self.client.infer("cifar_model", inputs=[inputs], outputs=[outputs])
            output = result.as_numpy("OUTPUT__0")
            predicted_class = np.argmax(output, axis=1)[0]
            
            print(f"🎯 Predição: {self.labels[predicted_class]}")
            print(f"📊 Probabilidades: {output[0]}")
            return True
        except Exception as e:
            print(f"❌ Servidor não disponível: {e}")
            return False

def main():
    print("🔮 Iniciando cliente de inference...")
    
    client = TritonInference()
    
    # Testar conexão
    if client.test_random():
        print("✅ Conectado ao Triton Server!")
        
        # Exemplo com imagem de teste (substitua por uma imagem real)
        print("📝 Para testar com uma imagem real:")
        print("   client.predict_image('caminho/para/sua/imagem.jpg')")
    else:
        print("❌ Não foi possível conectar ao Triton Server")
        print("💡 Certifique-se de que o servidor está rodando:")
        print("   docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 \\")
        print("   -v$(pwd)/model_repository:/models \\")
        print("   nvcr.io/nvidia/tritonserver:24.03-py3 \\")
        print("   tritonserver --model-repository=/models")

if __name__ == "__main__":
    main()