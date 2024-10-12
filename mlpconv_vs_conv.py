import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from mlpconv import MLPConv2d

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed()


# Define the standard ConvNet using Conv2d
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # Define a simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # (B, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (B, 256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 256, 8, 8)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (B, 512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 512, 4, 4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Define the MLPConvNet using MLPConv2d
class MLPConvNet(nn.Module):
    def __init__(self, num_classes=10, num_layers_per=2):
        super(MLPConvNet, self).__init__()
        # Define a similar architecture but replace some Conv2d layers with MLPConv2d
        self.features = nn.Sequential(
            MLPConv2d(3, 64, kernel_size=3, padding=1, num_layers=num_layers_per, hidden_size=512),  # (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 64, 16, 16)

            MLPConv2d(64, 128, kernel_size=3, padding=1, num_layers=num_layers_per, hidden_size=512),  # (B, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 128, 8, 8)

            MLPConv2d(128, 256, kernel_size=3, padding=1, num_layers=num_layers_per, hidden_size=512),  # (B, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 256, 4, 4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training and evaluation functions
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    t = time.time() - t
    print(f"Train Epoch: {epoch} \tLoss: {epoch_loss:.4f} \tAccuracy: {epoch_acc:.2f}%, Time: {t:.2f}s")
    return epoch_loss, epoch_acc

def evaluate(model, device, test_loader, criterion, epoch, mode='Validation'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    t = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    t = time.time() - t
    print(f"{mode} Epoch: {epoch} \tLoss: {epoch_loss:.4f} \tAccuracy: {epoch_acc:.2f}%, Time: {t:.2f}s")
    return epoch_loss, epoch_acc

# Main function to run the experiment
def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 20
    batch_size = 128
    learning_rate = 0.001

    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='../../datasets/CIFAR-10', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='../../datasets/CIFAR-10', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize models
    conv_model = ConvNet().to(device)
    mlpconv_model = MLPConvNet().to(device)

    # Print number of parameters
    conv_params = count_parameters(conv_model)
    mlpconv_params = count_parameters(mlpconv_model)
    print(f"ConvNet Parameters: {conv_params}")
    print(f"MLPConvNet Parameters: {mlpconv_params}")

    # Ensure roughly same number of parameters
    # Adjust MLPConvNet's hidden sizes if necessary
    # For simplicity, we proceed as defined

    # Define loss criterion
    criterion = nn.CrossEntropyLoss()

    # Define optimizers
    conv_optimizer = optim.Adam(conv_model.parameters(), lr=learning_rate)
    mlpconv_optimizer = optim.Adam(mlpconv_model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    conv_scheduler = optim.lr_scheduler.StepLR(conv_optimizer, step_size=10, gamma=0.1)
    mlpconv_scheduler = optim.lr_scheduler.StepLR(mlpconv_optimizer, step_size=10, gamma=0.1)

    # Containers to store metrics
    history = {
        'ConvNet': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
        'MLPConvNet': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
    }

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train ConvNet
        print("Training ConvNet:")
        train_loss, train_acc = train(conv_model, device, train_loader, conv_optimizer, criterion, epoch)
        val_loss, val_acc = evaluate(conv_model, device, test_loader, criterion, epoch, mode='Validation')
        history['ConvNet']['train_loss'].append(train_loss)
        history['ConvNet']['train_acc'].append(train_acc)
        history['ConvNet']['val_loss'].append(val_loss)
        history['ConvNet']['val_acc'].append(val_acc)
        conv_scheduler.step()

        # Train MLPConvNet
        print("\nTraining MLPConvNet:")
        train_loss, train_acc = train(mlpconv_model, device, train_loader, mlpconv_optimizer, criterion, epoch)
        val_loss, val_acc = evaluate(mlpconv_model, device, test_loader, criterion, epoch, mode='Validation')
        history['MLPConvNet']['train_loss'].append(train_loss)
        history['MLPConvNet']['train_acc'].append(train_acc)
        history['MLPConvNet']['val_loss'].append(val_loss)
        history['MLPConvNet']['val_acc'].append(val_acc)
        mlpconv_scheduler.step()

    # Save models
    os.makedirs('models', exist_ok=True)
    torch.save(conv_model.state_dict(), 'models/conv_model.pth')
    torch.save(mlpconv_model.state_dict(), 'models/mlpconv_model.pth')
    print("\nModels saved.")

    # Plot training curves
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['ConvNet']['train_loss'], label='ConvNet Train Loss')
    plt.plot(epochs, history['ConvNet']['val_loss'], label='ConvNet Val Loss')
    plt.plot(epochs, history['MLPConvNet']['train_loss'], label='MLPConvNet Train Loss')
    plt.plot(epochs, history['MLPConvNet']['val_loss'], label='MLPConvNet Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.annotate(f'ConvNet Params: {conv_params}', xy=(0.5, 0.01), xycoords='axes fraction', ha='center', fontsize=8)
    plt.annotate(f'MLPConvNet Params: {mlpconv_params}', xy=(0.5, 0.05), xycoords='axes fraction', ha='center', fontsize=8)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['ConvNet']['train_acc'], label='ConvNet Train Acc')
    plt.plot(epochs, history['ConvNet']['val_acc'], label='ConvNet Val Acc')
    plt.plot(epochs, history['MLPConvNet']['train_acc'], label='MLPConvNet Train Acc')
    plt.plot(epochs, history['MLPConvNet']['val_acc'], label='MLPConvNet Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.annotate(f'ConvNet Params: {conv_params}', xy=(0.5, 0.01), xycoords='axes fraction', ha='center', fontsize=8)
    plt.annotate(f'MLPConvNet Params: {mlpconv_params}', xy=(0.5, 0.05), xycoords='axes fraction', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # Final comparison
    print("\nFinal Results:")
    print(f"ConvNet - Validation Accuracy: {history['ConvNet']['val_acc'][-1]:.2f}% | Parameters: {conv_params}")
    print(f"MLPConvNet - Validation Accuracy: {history['MLPConvNet']['val_acc'][-1]:.2f}% | Parameters: {mlpconv_params}")

if __name__ == "__main__":
    main()
