# nvidia-ai-infrastructure

## Introdução à Infraestrutura de IA da NVIDIA (NCP-AI)

O design de infraestrutura de IA representa a base essencial para suportar cargas de trabalho de inteligência artificial em escala empresarial. Esta disciplina envolve a criação de ambientes computacionais especializados que integram hardware, software e recursos de rede otimizados especificamente para o ciclo de vida completo de projetos de ML e DL. Um design adequado precisa considerar desde o provisionamento de recursos acelerados para treinamento de modelos até a infraestrutura de inferência para deployment em produção, garantindo escalabilidade, confiabilidade e eficiência de custos ao longo de todo o processo.

**Pontos importantes complementares:**
- Composta por 5 camadas essenciais: computação, armazenamento, rede, software stack e orquestração
- **Computação**: inclui GPUs, CPUs e DPUs para offloading de tarefas como segurança e networking
- **Armazenamento**: NVMe para velocidade local e object stores para escalabilidade
- **Networking**: tecnologias críticas como NVLink e InfiniBand para mover grandes datasets
- **Orquestração**: Kubernetes é especialmente popular em infraestruturas de IA
- Cargas de trabalho são altamente dinâmicas - alguns jobs duram minutos, outros dias
- Princípios de design: escalabilidade, alta utilização de recursos, flexibilidade, segurança e monitorização

### Papel das GPUs em Cargas de Trabalho de IA
As GPUs emergiram como componentes críticos para IA devido à sua arquitetura massivamente paralela, ideal para processar as operações matriciais e de álgebra linear que fundamentam os algoritmos de deep learning. Diferente das CPUs com poucos núcleos otimizados para tarefas sequenciais, as GPUs possuem milhares de núcleos menores que processam simultaneamente grandes volumes de dados, reduzindo drasticamente o tempo necessário para treinar modelos complexos. Esta capacidade de aceleração paralela torna as GPUs indispensáveis para redes neurais profundas, processamento de linguagem natural e visão computacional.

**Pontos importantes complementares:**
- CPUs são limitadas para IA devido ao baixo número de núcleos e modelo de execução serial
- GPUs seguem arquitetura SIMD (Single Instruction Multiple Data) - ideal para processamento paralelo
- Memória de alta banda (HBM2, GDDR6) minimiza delays entre computação e acesso à memória
- **Benchmarks reais**: BERT em GPU é 20x mais rápido que CPU; inferência tem até 70% menor latência
- **ROI**: GPUs terminam jobs mais rápido, custo por modelo pode ser menor apesar do custo horário maior
- Frameworks como TensorFlow e PyTorch oferecem suporte nativo GPU via CUDA e cuDNN

### Arquiteturas CPU vs GPU vs DPU
As CPUs, GPUs e DPUs representam três paradigmas distintos de processamento complementares em infraestruturas de IA modernas. As CPUs funcionam como cérebro geral do sistema, gerenciando controle, lógica e tarefas sequenciais. As GPUs atuam como aceleradores especializados para computação paralela massiva em dados. Já as DPUs (Data Processing Units) são processadores inteligentes que descarregam tarefas de infraestrutura como networking, armazenamento e segurança, liberando CPUs e GPUs para focarem em cargas de trabalho de aplicação. Esta tríade forma a base da computação heterogênea contemporânea.

**Pontos importantes complementares:**
- **CPU**: "cérebro central" - executa OS, gerencia memória, orquestra jobs entre GPUs/TPUs
- **GPU**: "cavalos de força" da IA - milhares de núcleos leves para operações paralelas
- **DPU**: novo player - purpose-built para networking, storage, IO, segurança e telemetria
- DPUs permitem compartilhamento seguro de clusters GPU entre múltiplos usuários
- **Modelo dos três chips da NVIDIA**: CPU (orquestração), GPU (computação), DPU (infraestrutura)
- **Exemplo real**: Bluefield DPU combina 400Gb networking com aceleração criptográfica

### Aceleração por GPU para Pipelines de IA/ML
A aceleração por GPU revolucionou os pipelines de IA/ML ao otimizar cada etapa do fluxo de trabalho. Desde o pré-processamento de dados em larga escala até o treinamento iterativo de modelos e a inferência de alta throughput, as GPUs proporcionam ganhos de performance orders de magnitude superiores às soluções baseadas apenas em CPU. Esta aceleração permite experimentação mais rápida, redução do time-to-market e capacidade de lidar com datasets e modelos cada vez maiores e mais complexos, tornando viáveis aplicações que antes eram computacionalmente proibitivas.

**Pontos importantes complementares:**
- **Pré-processamento**: pode consumir 30-50% do tempo total do pipeline
- **RAPIDS**: fornece alternativas GPU para pandas, NumPy e scikit-learn (10-100x mais rápido)
- **DALI**: acelera pré-processamento de imagens/vídeo com operações no GPU
- **Treinamento distribuído**: Horovod, PyTorch DDP permitem escala em múltiplas GPUs/máquinas
- **Hyperparameter tuning**: GPUs permitem execução paralela de centenas de training runs
- **Inferência**: TensorRT para otimização + Triton para deployment escalável
- Fluxo completo acelerado reduz time-to-market e overhead operacional

### Visão Geral do Ecossistema NVIDIA (CUDA, Triton, NGC)
O ecossistema NVIDIA constitui uma plataforma abrangente e integrada para IA enterprise, centrada em três pilares principais: CUDA fornece o modelo de programação paralela que habilita a aceleração por GPU; Triton Inference Server oferece um ambiente unificado para deployment de modelos em produção com suporte a múltiplos frameworks e otimizações de performance; e o NGC (NVIDIA GPU Cloud) funciona como um catálogo de softwares, modelos pré-treinados e containers otimizados que aceleram o desenvolvimento e a implantação de soluções de IA. Juntos, esses componentes formam um stack coeso que simplifica a construção e operação de infraestruturas de IA escaláveis.

**Pontos importantes complementares:**
- **Stack completo**: drivers, bibliotecas, modelos pré-treinados, containers e ferramentas de orquestração
- **CUDA**: plataforma paralela que suporta C++, Python, Fortran - base de tudo
- **cuDNN**: operações de baixo nível para redes neurais (convoluções, pooling, ativações)
- **NGC**: hub central com Docker containers pré-configurados e modelos pré-treinados
- **Triton**: suporte multi-framework (TensorFlow, PyTorch, ONNX, XGBoost) com dynamic batching
- **DOCA**: SDK para programar DPUs - zero trust, multi-tenant isolation, observabilidade em tempo real
- Integração unificada permite portabilidade entre ambientes e performance máxima

## Gerenciamento de Recursos de GPU e Virtualização

### Configuração MIG (Multi-instance GPU)

A tecnologia **MIG (Multi-instance GPU)**, introduzida pela NVIDIA a partir da arquitetura Ampere (ex: GPUs A100, H100), permite particionar uma única GPU física em várias instâncias menores e totalmente isoladas no nível de hardware. Cada instância MIG possui seus próprios **Streaming Multiprocessors (SMs)**, **memória dedicada** (ex: 1GB, 5GB, 10GB) e **cache L2** alocados de forma fixa, operando de maneira independente e segura, como se fossem GPUs separadas. Isso é fundamental para **maximizar a utilização** em ambientes de *multi-tenancy* (como clouds, JupyterHub ou plataformas de inferência), onde múltiplos usuários ou cargas de trabalho (ex: diferentes modelos de IA) podem ser executados em paralelo na mesma GPU, sem risco de contenção de recursos ou interferência ("*noisy neighbors*"). A configuração é feita via comandos `nvidia-smi` (ex: `nvidia-smi mig -cgi 1g.5gb`) e pode ser gerenciada no Kubernetes via *device plugin* específico, permitindo alocação granular e previsível. O MIG é ideal para cenários que exigem **isolamento rigoroso** e eficiência de custos, transformando uma GPU poderosa (e muitas vezes subutilizada) em vários recursos menores e dedicados.

### Técnicas de Partilha e Isolamento de GPU

Para ambientes que não suportam MIG (GPUs mais antigas) ou necessitam de partilha mais dinâmica, existem técnicas de software para **partilha e isolamento de GPU**. A mais comum é o **Time-Slicing**, onde a GPU alterna entre processos em intervalos de tempo, sem isolamento físico – prático para desenvolvimento ou inferência leve, mas sujeito a imprevisibilidade de desempenho. Já a **isolação via contêineres** (Docker/ Kubernetes) com o **NVIDIA Container Toolkit** permite atribuir GPUs específicas a contêineres, usando *cgroups* para limitar recursos e evitar conflitos. Frameworks como TensorFlow e PyTorch oferecem controles de memória (ex: `tf.config.experimental.set_memory_growth`) para alocação consciente. No Kubernetes, o **NVIDIA Device Plugin** é essencial, expondo GPUs como recursos programáveis e permitindo o uso de *taints/tolerations* e *resource quotas* para agendamento justo. A escolha da técnica envolve trade-offs: Time-Slicing é simples mas menos isolado; contêineres oferecem lógica de isolamento; MIG garante separação física (mas requer hardware específico). A monitorização contínua com `nvidia-smi` ou **DCGM** é crucial para evitar saturação de memória e garantir *fairness*.

### Configuração e Casos de Uso de GPUs Virtuais (vGPU)

**GPUs Virtuais (vGPU)** da NVIDIA permitem virtualizar uma GPU física para ser partilhada por múltiplas **Máquinas Virtuais (VMs)**, cada uma recebendo uma fatia dedicada de computação e memória através do *hypervisor* (ex: VMware vSphere, Citrix Hypervisor, KVM). Diferente do MIG (focado em bare-metal/contêineres), o vGPUs é voltado para ambientes **virtualizados tradicionais**, sendo essencial para **VDI (Virtual Desktop Infrastructure)** – onde utilizadores remotos acedem a desktops com aceleração gráfica – e para infraestruturas de cloud que oferecem instâncias de VM com GPU. São definidos **perfis** (ex: vComputeServer para cargas computacionais, Quadro Virtual para gráficos) que determinam a quantidade de recursos alocados por VM. A configuração exige licenças específicas da NVIDIA, *drivers* compatíveis no host e nas VMs, e software de gestão (NVIDIA vGPU Software). Os casos de uso abrangem desde **estações de trabalho virtuais** para engenheiros de IA e designers gráficos até ambientes de **multi-inquilinato seguros** em universidades ou empresas, combinando a flexibilidade da virtualização (snapshots, migração) com o poder de processamento acelerado.

### Agendamento de Cargas de Trabalho com GPU no Kubernetes

O Kubernetes tornou-se a plataforma padrão para orquestrar cargas de trabalho aceleradas por GPU em escala. Para isso, é necessário instalar os **drivers NVIDIA** nos *nodes* e implantar o **NVIDIA Device Plugin** como um *DaemonSet*, que permite ao Kubernetes detetar e gerir GPUs como recursos programáveis (semelhante a CPU/memória). Nos *manifests* dos *Pods*, especifica-se o recurso `nvidia.com/gpu` sob `requests` e `limits` para garantir agendamento exclusivo. Técnicas avançadas como **nodeAffinity**, **taints/tolerations** e **resource quotas** ajudam a isolar *nodes* com GPU e a distribuir cargas de forma justa entre utilizadores ou equipas. Para GPUs com **MIG** (ex: A100), o *device plugin* suporta a exposição de instâncias individuais (ex: 1g.5gb) como recursos distintos, permitindo agendamento granular e multi-inquilinato seguro. A monitorização é feita com ferramentas como **NVIDIA DCGM** integrado com Prometheus/Grafana, fornecendo métricas detalhadas de utilização por *pod*. Melhores práticas incluem evitar misturar treino e inferência no mesmo *node*, usar *PriorityClasses* para cargas críticas e considerar *schedulers* avançados (ex: Volcano) para *batch jobs*. Esta abordagem permite gerir clusters de GPU de forma eficiente, resiliente e escalável.

### Laboratório Prático: Configurar MIG numa A100

Este laboratório prático oferece uma experiência hands-on para configurar a tecnologia **MIG numa GPU NVIDIA A100**. Os participantes aprenderão a ativar o modo MIG via `nvidia-smi`, a criar e gerir diferentes **perfis de instância** (ex: 1g.5gb, 2g.10gb, 3g.20gb) que dividem a GPU em partições isoladas, e a atribuir essas instâncias a contêineres ou cargas de trabalho específicas. O exercício inclui a verificação da configuração com comandos como `nvidia-smi mig -l` e a exploração de cenários reais, como a execução paralela de múltiplos modelos de inferência ou ambientes de desenvolvimento isolados na mesma GPU física. Este laboratório é essencial para compreender na prática como implementar **multi-inquilinato seguro e eficiente**, maximizando o retorno do investimento em hardware de última geração e preparando a infraestrutura para ambientes de produção escaláveis.

#### 📋 Pré-requisitos

- Sistema com **NVIDIA A100 GPU**
- Ubuntu 20.04+ (bare metal ou VM)
- Docker & Kubernetes (v1.20+)
- nvidia-container-toolkit, nvidia-docker2
- NVIDIA GPU Driver (465+)
- nvidia-smi, kubectl, helm

#### 🚀 Execução do Laboratório

Execute os scripts na ordem:

```bash
# 1. Habilitar modo MIG
chmod +x scripts/1-enable-mig.sh
./scripts/1-enable-mig.sh

# 2. Criar instâncias MIG
chmod +x scripts/2-create-mig-instances.sh
./scripts/2-create-mig-instances.sh

# 3. Implantar device plugin no Kubernetes
chmod +x scripts/3-deploy-device-plugin.sh
./scripts/3-deploy-device-plugin.sh

# 4. Testar com pod de exemplo
chmod +x scripts/4-deploy-test-pod.sh
./scripts/4-deploy-test-pod.sh

# Para monitoramento 
kubectl apply -f manifests/dcgm-exporter.yaml

```

As Instâncias MIG não são persistentes após reboot (use systemd para automação). Ajuste nodeSelector no mig-pod.yaml conforme seu ambiente e o comando ``nvidia-smi mig -lgip`` para ver perfis disponíveis.

## Armazenamento, Redes e Pipelines de Dados para IA

### Arquiteturas de Armazenamento para Cargas de Trabalho de IA (local, compartilhado, objeto)

O armazenamento na infraestrutura de IA é um componente crítico de desempenho, pois modelos de grande escala realizam leituras e escritas massivas de dados. Escolher a arquitetura correta — desde SSDs locais (NVMe) para velocidade, armazenamento compartilhado (como NFS ou Lustre) para treinamento distribuído, ou armazenamento de objeto (S3, GCS) para escalabilidade e custo — impacta diretamente a throughput, latência e utilização da GPU. Sistemas do mundo real frequentemente combinam esses tipos em uma arquitetura híbrida e em camadas (hot, warm, cold data) para otimizar custo e performance, garantindo que os GPUs nunca fiquem ociosos esperando por dados.

### Rede de Alta Velocidade: NVLink, Infiniband, RDMA

Em cargas de trabalho de IA distribuída, a rede é tão crucial quanto o poder de computação. Tecnologias como o NVLink da NVIDIA permitem comunicação ultrarrápida entre GPUs no mesmo nó, enquanto o InfiniBand é o padrão-ouro para interconexão de alta largura de banda e baixa latência entre nós em clusters. O RDMA (Remote Direct Memory Access) é fundamental, permitindo a transferência direta de dados entre a memória de máquinas diferentes, contornando a CPU e reduzindo drasticamente a latência e a sobrecarga. A combinação dessas tecnologias, juntamente com features como GPUDirect, é essencial para operações como "all-reduce" durante o treinamento distribuído de modelos grandes, como GPT ou BERT, garantindo que a sincronização de gradientes não se torne um gargalo.

### Gargalos e Otimização no Movimento de Dados

Um gargalo no movimento de dados — seja em E/S de disco, na rede ou no pré-processamento — pode deixar GPUs caros ociosas, aumentando o tempo de treinamento e custos operacionais. Identificar esses pontos é o primeiro passo, utilizando ferramentas como \texttt{iostat}, \texttt{nvtop} ou profilers de framework (TensorFlow, PyTorch). A otimização envolve estratégias como a adoção de NVMe, uso de carregamento de dados multi-thread (ex: \texttt{num\_workers} no PyTorch), prefetching, caching de dados localmente e paralelização do pré-processamento (ex: com NVIDIA DALI). Em nível de rede, tuning de configurações (MTU, buffers) e a adoção de InfiniBand com RDMA são chave para um fluxo de dados contínuo e eficiente do storage até a GPU.

### Design de Pipeline de Dados para IA (ETL + Treinamento + Inferência)

Um pipeline de dados de IA bem projetado é um sistema interconectado que abrange desde a ingestão de dados brutos (ETL) até o treinamento e a inferência. O estágio de ETL, frequentemente orquestrado por ferramentas como Apache Airflow e acelerado por GPUs (RAPIDS, DALI), é responsável por extrair, transformar e carregar dados em um storage acessível. No treinamento, o pipeline deve alimentar os GPUs de forma contínua, usando data loaders paralelizados e formatos eficientes. Para inferência, em lote ou tempo real, servidores de modelo como o Triton Inference Server são utilizados para oferecer baixa latência e alto throughput. Projetar com resiliência, monitoramento e estágios desacoplados (usando filas como Kafka) garante um pipeline robusto e escalável. 

### Laboratório: Projetar um Pipeline de Dados de Ponta a Ponta para IA

Neste laboratório prático, consolidamos todos os conceitos anteriores para projetar e implementar um pipeline completo. Isso envolve a configuração de uma arquitetura de armazenamento em camadas (ex: S3 para dados brutos, BeeGFS/Lustre para datasets de treinamento), a configuração de rede de alta velocidade (InfiniBand com RDMA) e a construção do fluxo de dados em si. Você poderá orquestrar um pipeline que ingere dados de um stream em tempo real (Kafka), realiza ETL acelerada, treina um modelo em um cluster de GPUs interconectados com NVLink/InfiniBand e, finalmente, implanta o modelo para inferência em um ambiente escalável como Kubernetes, utilizando otimizações para evitar gargalos e garantir a máxima utilização dos recursos.

#### 🎯 Objetivo

Criar um pipeline completo de IA demonstrando:
- **ETL** com NVIDIA DALI
- **Treinamento** com PyTorch
- **Inference** com Triton Server
- **Otimização** com GPU

#### 🛠️ Pré-requisitos

- NVIDIA GPU (A100, V100, RTX 3090, etc)
- Docker e NVIDIA Container Toolkit
- Python 3.8+

#### ⚡ Configuração Rápida

```bash
# 1. Clonar repositório
git clone https://github.com/seu-usuario/lab-ai-pipeline.git
cd lab-ai-pipeline

# 2. Executar setup automático
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# 3. Ativar ambiente virtual
source venv/bin/activate

```

Data Source → ETL (DALI) → Training (PyTorch) → Model → Triton Server → Inference Client

Treinar o modelo:

```
python src/train_model.py
```

Iniciar o servidor triton:

```bash
# Usando Docker Compose (recomendado)
docker-compose up triton-server
```

```bash
# Ou manualmente
docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v$(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.03-py3 \
  tritonserver --model-repository=/models
```

Testar inferencia:

```bash
python src/inference_client.py
```

Jupyternotebook:
```bash
docker-compose up jupyter
# Acesse: http://localhost:8888
```

Benchmark:

```bash
# Instalar perf_analyzer
docker exec -it <triton_container> perf_analyzer -m my_model -b 8 -u localhost:8000
```

Monitoramento: 

```bash
watch -n 1 nvidia-smi
```

####Versão Simplificada

Pipeline completo ETL → Treinamento → Inference em 3 arquivos!

## ⚡ Comece Agora

```bash
# 1. Clone e instale
git clone <seu-repositorio>
cd lab-ai-pipeline
pip install -r requirements.txt

# 2. Treine o modelo
python train.py

# 3. Inicie o servidor (terminal 1)
docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:24.03-py3 tritonserver --model-repository=/models

# 4. Teste inference (terminal 2)
python inference.py
```

## Orquestração e Escalabilidade de Clusters de IA

## Otimização de Desempenho e Monitoramento

## Segurança, Conformidade e Governança de Dados

## Infraestrutura de IA na Edge e Integração

## NGC, Triton Inference Server e Implantação

## Projetos do Mundo Real e Fluxos de Trabalho Empresariais

## Projeto Final e Preparação para Certificação
