#!/bin/bash

set -e

echo "🚀 Implantando Lab 6 - Políticas de Segurança em Infraestrutura de IA"
echo "======================================================================"

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Verificar pré-requisitos
echo -e "${BLUE}🔍 Verificando pré-requisitos...${NC}"

# Verificar kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${YELLOW}❌ kubectl não encontrado${NC}"
    exit 1
fi

# Verificar cluster
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${YELLOW}❌ Não é possível conectar ao cluster Kubernetes${NC}"
    exit 1
fi

# Verificar se é cluster-admin
if ! kubectl auth can-i '*' '*' --all-namespaces &> /dev/null; then
    echo -e "${YELLOW}❌ Acesso cluster-admin necessário${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pré-requisitos verificados${NC}"

# Função para aplicar manifests
apply_manifests() {
    local dir=$1
    local desc=$2
    
    echo -e "${BLUE}📦 Aplicando $desc...${NC}"
    if [ -d "$dir" ] && [ "$(ls -A $dir)" ]; then
        kubectl apply -f $dir/
        echo -e "${GREEN}✅ $desc aplicado${NC}"
    else
        echo -e "${YELLOW}⚠️  Nenhum manifest encontrado em $dir${NC}"
    fi
}

# 1. Criar namespaces seguros
echo -e "\n${BLUE}📍 1. Criando namespaces seguros...${NC}"
kubectl apply -f manifests/00-namespaces/

# 2. Configurar RBAC
apply_manifests "manifests/01-rbac" "RBAC"

# 3. Configurar quotas e limites de recursos
apply_manifests "manifests/02-resources" "Resource Quotas"

# 4. Configurar políticas de rede
apply_manifests "manifests/03-network" "Network Policies"

# 5. Gerar e configurar secrets TLS
echo -e "\n${BLUE}🔐 5. Gerando certificados TLS...${NC}"
./scripts/generate-certs.sh
apply_manifests "manifests/04-secrets" "TLS Secrets"

# 6. Implantar workloads seguros
apply_manifests "manifests/05-workloads" "Secure Workloads"

# 7. Configurar admission control (opcional)
read -p "❓ Deseja instalar Gatekeeper para admission control? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🛡️  Instalando Gatekeeper...${NC}"
    
    # Instalar Gatekeeper
    kubectl apply -f https://raw.githubusercontent.com/open-policy-agent/gatekeeper/master/deploy/gatekeeper.yaml
    
    # Aguardar inicialização
    echo "⏳ Aguardando Gatekeeper inicializar..."
    sleep 30
    
    apply_manifests "manifests/06-admission" "Admission Policies"
fi

# 8. Configurar monitoramento (opcional)
read -p "❓ Deseja configurar monitoramento de GPU? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}📊 Configurando monitoramento...${NC}"
    apply_manifests "manifests/07-monitoring" "Monitoring"
fi

echo -e "\n${GREEN}✅ Implantação concluída!${NC}"
echo -e "\n${YELLOW}📝 Próximos passos:${NC}"
echo -e "   ./scripts/validate-controls.sh    # Validar controles de segurança"
echo -e "   kubectl get pods -n team-a        # Verificar workloads"
echo -e "   kubectl get networkpolicies -A    # Verificar políticas de rede"