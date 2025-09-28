import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class SimpleCNN(nn.Module):
    """Modelo CNN simples para classificação CIFAR-10"""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    """Função principal de treinamento"""
    
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Transformações
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Dataset e DataLoader
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
        num_workers=4
    )
    
    # Modelo, otimizador e loss
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Loop de treinamento
    losses = []
    for epoch in range(5):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader, 0):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 200 == 199:
                print(f'Epoch [{epoch + 1}/5], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 200:.4f}')
                losses.append(running_loss / 200)
                running_loss = 0.0
    
    # Salvar modelo
    os.makedirs('../model_repository/my_model/1', exist_ok=True)
    torch.save(model.state_dict(), '../model_repository/my_model/1/model.pt')
    
    print("Treinamento concluído! Modelo salvo.")
    return losses

if __name__ == "__main__":
    losses = train_model()
    
    # Plotar perda
    plt.plot(losses)
    plt.title('Perda durante o treinamento')
    plt.xlabel('Batch (x200)')
    plt.ylabel('Loss')
    plt.savefig('../training_loss.png')
    plt.show()