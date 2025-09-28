import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Modelo CNN simples para CIFAR-10
class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

def main():
    print("🚀 Iniciando treinamento do modelo CIFAR-10...")
    
    # Configuração
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 Usando dispositivo: {device}")
    
    # Transformações dos dados
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Carregar dataset
    print("📥 Baixando CIFAR-10...")
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        num_workers=2
    )
    
    # Modelo e otimizador
    model = CIFARModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("🎯 Iniciando treinamento...")
    
    # Loop de treinamento
    for epoch in range(3):  # 3 épocas para teste rápido
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'📊 Época {epoch+1}/3, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Salvar modelo
    os.makedirs('model_repository/cifar_model/1', exist_ok=True)
    torch.save(model.state_dict(), 'model_repository/cifar_model/1/model.pt')
    
    print("✅ Treinamento concluído! Modelo salvo em model_repository/cifar_model/1/model.pt")

if __name__ == "__main__":
    main()