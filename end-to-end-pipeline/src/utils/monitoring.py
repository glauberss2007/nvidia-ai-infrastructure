import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch

def monitor_gpu(duration=60, interval=2):
    """Monitora a utilização da GPU"""
    print(f"📊 Monitoring GPU for {duration} seconds...")
    
    metrics = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Executa nvidia-smi
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    timestamp, index, gpu_util, mem_used, mem_total, temp = line.split(', ')
                    metrics.append({
                        'timestamp': timestamp,
                        'gpu_index': int(index),
                        'gpu_utilization': int(gpu_util),
                        'memory_used': int(mem_used),
                        'memory_total': int(mem_total),
                        'temperature': int(temp)
                    })
                    print(f"GPU {index}: {gpu_util}% util, {mem_used}/{mem_total} MB, {temp}°C")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("Monitoring interrupted")
    
    # Cria relatório
    if metrics:
        df = pd.DataFrame(metrics)
        print("\n📈 GPU Monitoring Report:")
        print(f"   Average GPU Utilization: {df['gpu_utilization'].mean():.1f}%")
        print(f"   Average Memory Usage: {df['memory_used'].mean():.1f} MB")
        print(f"   Average Temperature: {df['temperature'].mean():.1f}°C")
        
        return df
    return None

def benchmark_model(model, test_loader, device='cuda'):
    """Benchmark de performance do modelo"""
    model.eval()
    model.to(device)
    
    print("🧪 Running model benchmark...")
    
    # Warm-up
    print("   Warm-up...")
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= 10:  # 10 batches de warm-up
                break
            _ = model(images.to(device))
    
    # Benchmark
    print("   Benchmarking...")
    start_time = time.time()
    total_samples = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
            total_samples += images.size(0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    throughput = total_samples / total_time
    
    print(f"📊 Benchmark Results:")
    print(f"   Total samples: {total_samples}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Throughput: {throughput:.2f} samples/second")
    
    return throughput