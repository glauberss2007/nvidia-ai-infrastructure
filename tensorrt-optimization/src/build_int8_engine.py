#!/usr/bin/env python3
"""
Script para construir engine TensorRT INT8 com calibração
"""

import os
import glob
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image

# Logger do TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class ImageBatchStream:
    def __init__(self, batch_size, calib_dir, shape=(3, 224, 224)):
        self.batch_size = batch_size
        self.files = glob.glob(os.path.join(calib_dir, '*.jpg')) + glob.glob(os.path.join(calib_dir, '*.png'))
        self.shape = shape
        self.index = 0
        
        if len(self.files) == 0:
            raise ValueError(f"Nenhuma imagem encontrada em {calib_dir}")
            
        print(f"🔍 Encontradas {len(self.files)} imagens para calibração")

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index + self.batch_size > len(self.files):
            return None
            
        batch_files = self.files[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        batch = []
        
        for f in batch_files:
            try:
                img = Image.open(f).convert('RGB').resize((224, 224))
                arr = np.asarray(img).astype(np.float32) / 255.0
                arr = (arr - 0.5) / 0.5  # Normalização simples
                arr = np.transpose(arr, (2, 0, 1))  # CHW
                batch.append(arr)
            except Exception as e:
                print(f"⚠️  Erro ao processar {f}: {e}")
                continue
                
        return np.ascontiguousarray(batch) if batch else None

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batchstream, cache_file="calib.cache"):
        super().__init__()
        self.stream = batchstream
        self.d_input = cuda.mem_alloc(trt.volume((batchstream.batch_size, *batchstream.shape)) * 4)  # 4 bytes per float32
        self.cache_file = cache_file
        self.stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        batch = self.stream.next_batch()
        if batch is None:
            return None
            
        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_int8_engine(onnx_path, engine_path, calib_dir, batch_size=16):
    print(f"🔨 Construindo engine INT8: {onnx_path} -> {engine_path}")
    
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Falha no parsing ONNX:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    if not builder.platform_has_fast_int8:
        raise RuntimeError("INT8 não suportado nesta GPU")
        
    config.set_flag(trt.BuilderFlag.INT8)
    
    # Permitir fallback para FP16 se necessário
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Perfil de otimização (batch dinâmico)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 3, 224, 224),
        opt=(8, 3, 224, 224),
        max=(32, 3, 224, 224)
    )
    config.add_optimization_profile(profile)

    # Calibrador
    calibrator = EntropyCalibrator(ImageBatchStream(batch_size, calib_dir))
    config.int8_calibrator = calibrator

    # Construir engine
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Falha ao construir engine")
        
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
        
    print(f"✅ Engine INT8 salva: {engine_path}")

def main():
    onnx_path = "../models/resnet50.onnx"
    engine_path = "../models/resnet50_int8.plan"
    calib_dir = "../calibration"
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Arquivo ONNX não encontrado: {onnx_path}")
        
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)
        print(f"⚠️  Diretório de calibração criado: {calib_dir}")
        print("   Por favor, adicione imagens JPEG/PNG para calibração")
        return
        
    build_int8_engine(onnx_path, engine_path, calib_dir)

if __name__ == "__main__":
    main()