import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
import time
import os

class TritonClient:
    def __init__(self, url="localhost:8080"):
        self.client = httpclient.InferenceServerClient(url=url)
        
    def check_server_ready(self):
        """Verifica se o servidor Triton está pronto"""
        try:
            return self.client.is_server_ready()
        except Exception as e:
            print(f"❌ Triton server not ready: {e}")
            return False
    
    def check_model_ready(self, model_name="my_model"):
        """Verifica se o modelo está pronto"""
        try:
            return self.client.is_model_ready(model_name)
        except Exception as e:
            print(f"❌ Model not ready: {e}")
            return False
    
    def inference(self, model_name, input_data):
        """Executa inferência no modelo"""
        
        # Prepara a entrada
        inputs = httpclient.InferInput("INPUT__0", input_data.shape, "FP32")
        inputs.set_data_from_numpy(input_data.astype(np.float32))
        
        # Prepara a saída
        outputs = httpclient.InferRequestedOutput("OUTPUT__0")
        
        # Executa inferência
        result = self.client.infer(model_name, inputs=[inputs], outputs=[outputs])
        
        return result.as_numpy("OUTPUT__0")
    
    def benchmark_model(self, model_name, input_shape=(1, 3, 32, 32), iterations=100):
        """Benchmark do modelo com dados aleatórios"""
        print(f"🧪 Running benchmark for {model_name}...")
        
        total_time = 0
        for i in range(iterations):
            # Gera dados aleatórios
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # Executa inferência
            start_time = time.time()
            result = self.inference(model_name, input_data)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            
            if i % 20 == 0:
                print(f"   Iteration {i}: {result.shape} - {end_time - start_time:.4f}s")
        
        avg_time = total_time / iterations
        print(f"📊 Average inference time: {avg_time:.4f}s")
        print(f"📊 Throughput: {1/avg_time:.2f} inferences/second")
        
        return avg_time

def create_model_repository():
    """Cria a estrutura do repositório de modelos do Triton"""
    import os
    
    # Estrutura de diretórios
    directories = [
        "src/inference/model_repository/my_model/1",
        "src/inference/model_repository/my_model"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Model repository structure created")

def test_triton_connection():
    """Testa a conexão com o Triton Server"""
    client = TritonClient()
    
    if client.check_server_ready():
        print("✅ Triton server is ready!")
        
        if client.check_model_ready("my_model"):
            print("✅ Model 'my_model' is ready!")
            
            # Testa inferência
            test_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
            result = client.inference("my_model", test_input)
            print(f"✅ Test inference successful! Output shape: {result.shape}")
            
            return True
        else:
            print("❌ Model 'my_model' is not ready")
            return False
    else:
        print("❌ Triton server is not ready")
        return False