#!/bin/bash
echo "=== Iniciando Triton Inference Server ==="

docker run --gpus=all --rm -it \
    -p8080:8080 -p8081:8081 -p8082:8082 \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/tritonserver:23.12-py3 \
    tritonserver --model-repository=/models

echo "Triton executando em:"
echo "HTTP: localhost:8080"
echo "gRPC: localhost:8081"
echo "Metrics: localhost:8082"