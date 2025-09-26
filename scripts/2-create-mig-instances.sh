#!/bin/bash
echo "=== Step 2: Create MIG Instances ==="

# List available profiles
echo "Available MIG profiles:"
nvidia-smi mig -lgip

# Create 3 instances of 2g.10gb profile (profile ID 0)
echo "Creating 3 instances of 2g.10gb profile..."
sudo nvidia-smi mig -cgi 0 -gi 3 -i 0

# Verify created instances
echo "Verifying MIG instances:"
nvidia-smi -L

echo "MIG instances created successfully."