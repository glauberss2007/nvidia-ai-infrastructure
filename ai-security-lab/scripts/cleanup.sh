#!/bin/bash

set -e

echo "🧹 Limpando ambiente do Lab 6..."

# Confirmar com usuário
read -p "❓ Tem certeza que deseja limpar todo o ambiente do Lab 6? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operação cancelada."
    exit 0
fi

# Deletar namespaces (isso deleta todos os recursos dentro deles)
echo "🗑️  Deletando namespaces..."
kubectl delete namespace team-a team-b --ignore-not-found=true

# Deletar recursos do Gatekeeper
echo "🛡️  Limpando Gatekeeper..."
kubectl delete constrainttemplate k8sapprovedimages --ignore-not-found=true
kubectl delete K8sApprovedImages only-approved-registries --ignore-not-found=true

# Deletar recursos de monitoramento
echo "📊 Limpando monitoramento..."
kubectl delete prometheusrule gpu-security-alerts -n monitoring --ignore-not-found=true
kubectl delete daemonset dcgm-exporter -n monitoring --ignore-not-found=true
kubectl delete configmap dcgm-exporter-config -n monitoring --ignore-not-found=true

# Deletar arquivos locais
echo "📁 Limpando arquivos locais..."
rm -rf ../manifests/04-secrets/*.pem

echo -e "\n✅ Limpeza concluída!"
echo -e "\n${YELLOW}📝 Nota:${NC}"
echo "O Gatekeeper e Prometheus não foram removidos do cluster, apenas as políticas específicas."
echo "Para remover completamente:"
echo "  kubectl delete -f https://raw.githubusercontent.com/open-policy-agent/gatekeeper/master/deploy/gatekeeper.yaml"