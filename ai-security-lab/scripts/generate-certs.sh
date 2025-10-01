#!/bin/bash

set -e

echo "🔐 Gerando certificados TLS para segurança..."

mkdir -p ../manifests/04-secrets

# Gerar certificado auto-assinado
openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
  -keyout ../manifests/04-secrets/key.pem \
  -out ../manifests/04-secrets/cert.pem \
  -subj "/CN=triton.team-a.svc.cluster.local"

echo "✅ Certificados gerados:"
echo "   - ../manifests/04-secrets/cert.pem"
echo "   - ../manifests/04-secrets/key.pem"