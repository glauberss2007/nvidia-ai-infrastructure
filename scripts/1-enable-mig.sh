#!/bin/bash
echo "=== Step 1: Enable MIG Mode on A100 GPU ==="

# Enable MIG on GPU 0
sudo nvidia-smi -i 0 -mig 1

# Enable on additional GPUs if present
# sudo nvidia-smi -i 1 -mig 1
# sudo nvidia-smi -i 2 -mig 1

echo "MIG mode enabled. System reboot may be required if this is the first time."
echo "Check status with: nvidia-smi"