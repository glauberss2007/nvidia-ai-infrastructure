#!/usr/bin/env python3
"""
Cliente para validação de correção entre diferentes versões do modelo
"""

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import *
import torch
import torch.nn.functional as F
from PIL import Image
import glob
import os

class TritonValidator:
    def __init__(self, url="localhost:8001"):
        self.client = grpcclient.InferenceServerClient(url=url)
        
    def preprocess_image(self, image_path):
        """Pré-processamento de imagem para ResNet50"""
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.asarray(img).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5  # Normalização
        img_array = np.transpose(img_array, (2, 0, 1))  # CHW
        return img_array
    
    def get_predictions(self, model_name, image_batch):
        """Obter predições do modelo"""
        inputs = [
            grpcclient.InferInput("input", image_batch.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(image_batch)
        
        outputs = [grpcclient.InferRequestedOutput("logits")]
        
        response = self.client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        return response.as_numpy("logits")
    
    def compare_models(self, test_images, num_images=5):
        """Comparar predições entre diferentes versões do modelo"""
        
        models = ["resnet50_fp32"]
        if self.client.is_model_ready("resnet50_trt_fp16"):
            models.append("resnet50_trt_fp16")
        if self.client.is_model_ready("resnet50_trt_int8"):
            models.append("resnet50_trt_int8")
        
        print(f"🔍 Comparando modelos: {models}")
        
        results = {}
        
        for i, img_path in enumerate(test_images[:num_images]):
            print(f"\n📊 Imagem {i+1}: {os.path.basename(img_path)}")
            
            # Pré-processar imagem
            img_array = self.preprocess_image(img_path)
            batch = np.expand_dims(img_array, axis=0)
            
            # Obter predições de cada modelo
            predictions = {}
            for model in models:
                logits = self.get_predictions(model, batch)
                probs = torch.softmax(torch.tensor(logits), dim=1)
                top5_probs, top5_indices = torch.topk(probs, 5)
                
                predictions[model] = {
                    'logits': logits,
                    'top5_indices': top5_indices.numpy(),
                    'top5_probs': top5_probs.numpy()
                }
                
                print(f"  {model}:")
                print(f"    Top-1: {top5_indices[0][0]} ({top5_probs[0][0]:.4f})")
            
            # Comparar com baseline (FP32)
            baseline_top1 = predictions['resnet50_fp32']['top5_indices'][0][0]
            
            for model in models[1:]:
                model_top1 = predictions[model]['top5_indices'][0][0]
                match = "✅" if model_top1 == baseline_top1 else "❌"
                print(f"    {match} {model} vs FP32: {'MATCH' if model_top1 == baseline_top1 else 'MISMATCH'}")
                
                # Calcular similaridade cosseno nos logits
                cos_sim = self.cosine_similarity(
                    predictions['resnet50_fp32']['logits'].flatten(),
                    predictions[model]['logits'].flatten()
                )
                print(f"    Similaridade cosseno: {cos_sim:.6f}")
        
        return results
    
    def cosine_similarity(self, a, b):
        """Calcular similaridade cosseno entre dois vetores"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validar correção entre modelos')
    parser.add_argument('--image-dir', default='../calibration', help='Diretório com imagens de teste')
    parser.add_argument('--num-images', type=int, default=3, help='Número de imagens para teste')
    
    args = parser.parse_args()
    
    # Encontrar imagens de teste
    test_images = glob.glob(os.path.join(args.image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(args.image_dir, "*.png"))
    
    if not test_images:
        print("❌ Nenhuma imagem encontrada para teste")
        print("   Coloque imagens JPEG/PNG em ../calibration/")
        return
    
    print(f"🔍 Encontradas {len(test_images)} imagens para validação")
    
    # Validar
    validator = TritonValidator()
    validator.compare_models(test_images, args.num_images)

if __name__ == "__main__":
    main()