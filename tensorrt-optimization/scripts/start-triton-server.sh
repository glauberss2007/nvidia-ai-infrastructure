#!/bin/bash

set -e

echo "🚀 Iniciando Triton Inference Server..."

# Criar repositório de modelos Triton
mkdir -p ../model_repository

# Configurações dos modelos
setup_model_repository() {
    echo "📁 Configurando repositório de modelos..."
    
    # Modelo FP32 (ONNX)
    mkdir -p ../model_repository/resnet50_fp32/1
    if [ -f "../models/resnet50.onnx" ]; then
        cp ../models/resnet50.onnx ../model_repository/resnet50_fp32/1/model.onnx
    fi
    
    cat > ../model_repository/resnet50_fp32/config.pbtxt << 'EOF'
name: "resnet50_fp32"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
    { name: "input" data_type: TYPE_FP32 dims: [3, 224, 224] }
]
output [
    { name: "logits" data_type: TYPE_FP32 dims: [1000] }
]

dynamic_batching {
    preferred_batch_size: [ 8, 16, 32 ]
    max_queue_delay_microseconds: 1000
}

instance_group [{ kind: KIND_GPU, count: 2 }]
EOF

    # Modelo FP16 (TensorRT)
    mkdir -p ../model_repository/resnet50_trt_fp16/1
    if [ -f "../models/resnet50_fp16.plan" ]; then
        cp ../models/resnet50_fp16.plan ../model_repository/resnet50_trt_fp16/1/model.plan
    fi
    
    cat > ../model_repository/resnet50_trt_fp16/config.pbtxt << 'EOF'
name: "resnet50_trt_fp16"
platform: "tensorrt_plan"
max_batch_size: 32
input [
    { name: "input" data_type: TYPE_FP32 dims: [3, 224, 224] }
]
output [
    { name: "logits" data_type: TYPE_FP32 dims: [1000] }
]

optimization {
    execution_accelerators {
        gpu_execution_accelerator : [ { name : "tensorrt" } ]
    }
}

dynamic_batching {
    preferred_batch_size: [ 8, 16, 32 ]
    max_queue_delay_microseconds: 800
}

instance_group [{ kind: KIND_GPU, count: 2 }]
EOF

    # Modelo INT8 (TensorRT) - se existir
    if [ -f "../models/resnet50_int8.plan" ]; then
        mkdir -p ../model_repository/resnet50_trt_int8/1
        cp ../models/resnet50_int8.plan ../model_repository/resnet50_trt_int8/1/model.plan
        
        cat > ../model_repository/resnet50_trt_int8/config.pbtxt << 'EOF'
name: "resnet50_trt_int8"
platform: "tensorrt_plan"
max_batch_size: 32
input [{ name: "input" data_type: TYPE_FP32 dims: [3, 224, 224] }]
output [{ name: "logits" data_type: TYPE_FP32 dims: [1000] }]

dynamic_batching { 
    preferred_batch_size: [8, 16, 32] 
    max_queue_delay_microseconds: 800 
}

instance_group [{ kind: KIND_GPU, count: 2 }]
EOF
    fi
}

setup_model_repository

echo "🔍 Verificando modelos no repositório:"
find ../model_repository -name "*.onnx" -o -name "*.plan" | sort

# Iniciar servidor Triton
echo "🌐 Iniciando Triton Server na porta 8000-8002..."
docker run -d --name triton-server --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/../model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.03-py3 \
    tritonserver --model-repository=/models --log-verbose=1

echo "⏳ Aguardando servidor inicializar..."
sleep 10

# Verificar status
echo "🔍 Verificando status do servidor..."
docker logs triton-server --tail 10

echo "✅ Triton Server iniciado!"
echo "📊 Endpoints:"
echo "   HTTP: localhost:8000"
echo "   gRPC: localhost:8001"
echo "   Metrics: localhost:8002"