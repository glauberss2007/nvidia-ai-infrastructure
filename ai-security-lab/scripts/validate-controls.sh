#!/bin/bash

set -e

echo "🧪 Validando controles de segurança do Lab 6"
echo "============================================"

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅ $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️ $1${NC}"; }

# 1. Validar namespaces e labels de segurança
echo -e "\n🔍 1. Validando namespaces e segurança de pods..."
if kubectl get ns team-a team-b &> /dev/null; then
    pass "Namespaces team-a e team-b criados"
    
    # Verificar labels de segurança
    if kubectl get ns team-a -o jsonpath='{.metadata.labels}' | grep -q "restricted"; then
        pass "Label de segurança restrita no namespace team-a"
    else
        fail "Label de segurança restrita não encontrada no team-a"
    fi
else
    fail "Namespaces não encontrados"
fi

# 2. Validar RBAC
echo -e "\n🔐 2. Validando RBAC..."
kubectl -n team-a auth can-i list secrets --as=system:serviceaccount:team-a:ds-runner > /dev/null 2>&1
if [ $? -ne 0 ]; then
    pass "RBAC: Service account não pode listar secrets (conforme esperado)"
else
    fail "RBAC: Service account pode listar secrets (INSEGURO!)"
fi

# 3. Validar quotas de GPU
echo -e "\n📊 3. Validando quotas de recursos..."
if kubectl -n team-a get resourcequota rq-gpu-and-objects &> /dev/null; then
    pass "ResourceQuota para GPU configurado"
else
    fail "ResourceQuota para GPU não encontrado"
fi

# 4. Validar NetworkPolicies
echo -e "\n🌐 4. Validando NetworkPolicies..."
if kubectl -n team-a get networkpolicy default-deny-all &> /dev/null; then
    pass "NetworkPolicy default-deny configurada"
else
    fail "NetworkPolicy default-deny não encontrada"
fi

# 5. Validar TLS Secrets
echo -e "\n🔒 5. Validando Secrets TLS..."
if kubectl -n team-a get secret triton-tls &> /dev/null; then
    pass "Secret TLS criado"
else
    fail "Secret TLS não encontrado"
fi

# 6. Validar segurança de workloads
echo -e "\n🐳 6. Validando segurança de workloads..."
echo "📋 Testando pod privilegiado (deve ser bloqueado)..."
kubectl -n team-a apply -f - <<'YAML' > /dev/null 2>&1
apiVersion: v1
kind: Pod
metadata: { name: test-privileged }
spec:
    containers:
    - name: test
      image: busybox
      command: ["sh", "-c", "sleep 3600"]
      securityContext:
        privileged: true
YAML

sleep 2

if kubectl -n team-a get pod test-privileged &> /dev/null; then
    fail "Pod privilegiado foi criado (deveria ser bloqueado)"
    kubectl -n team-a delete pod test-privileged --force > /dev/null 2>&1
else
    pass "Pod Security Standards bloquearam pod privilegiado"
fi

# 7. Validar workloads seguros
echo -e "\n🔍 7. Verificando workloads seguros..."
if kubectl -n team-a get job cuda-secure-job &> /dev/null; then
    pass "Job seguro criado"
    
    # Verificar securityContext
    if kubectl -n team-a get job cuda-secure-job -o yaml | grep -q "runAsNonRoot: true"; then
        pass "Workload configurado como non-root"
    else
        warn "Workload não está configurado como non-root"
    fi
else
    fail "Job seguro não encontrado"
fi

# 8. Validar Gatekeeper (se instalado)
echo -e "\n🛡️ 8. Validando Admission Control..."
if kubectl get deployment gatekeeper-controller-manager -n gatekeeper-system &> /dev/null; then
    pass "Gatekeeper instalado"
    
    # Testar política de registries
    echo "📋 Testando política de registries aproved..."
    kubectl -n team-a apply -f - <<'YAML' > /dev/null 2>&1
apiVersion: v1
kind: Pod
metadata: { name: test-unapproved-image }
spec:
    containers:
    - name: test
      image: docker.io/library/ubuntu:latest
      command: ["sleep", "3600"]
YAML

    sleep 2
    if kubectl -n team-a get pod test-unapproved-image &> /dev/null; then
        fail "Pod com imagem não aprovada foi criado"
        kubectl -n team-a delete pod test-unapproved-image --force > /dev/null 2>&1
    else
        pass "Gatekeeper bloqueou imagem não aprovada"
    fi
else
    warn "Gatekeeper não instalado - pulando validação de admission control"
fi

echo -e "\n${GREEN}🎉 Validação concluída!${NC}"
echo -e "\n${YELLOW}📝 Resumo:${NC}"
echo "Foram validadas as seguintes camadas de segurança:"
echo "  ✅ Namespaces seguros"
echo "  ✅ RBAC de menor privilégio" 
echo "  ✅ Quotas de GPU"
echo "  ✅ NetworkPolicies"
echo "  ✅ TLS Secrets"
echo "  ✅ Pod Security Standards"
echo "  ✅ Workloads seguros"
echo "  ✅ Admission Control (se instalado)"