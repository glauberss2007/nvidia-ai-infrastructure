import requests
import json

# Client exemplo para fazer inferência
def query_model(model_name, input_data):
    url = f"http://localhost:8080/v2/models/{model_name}/infer"
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=input_data, headers=headers)
    return response.json()

# Exemplo de uso
if __name__ == "__main__":
    with open('sample_request.json', 'r') as f:
        sample_data = json.load(f)
    
    # Testar ambos os modelos
    print("Testing TensorFlow model:")
    result_tf = query_model("tf_model", sample_data)
    print(result_tf)
    
    print("\nTesting ONNX model:")
    result_onnx = query_model("onnx_model", sample_data)
    print(result_onnx)