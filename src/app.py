import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import mlflow
from mlflow import pytorch
import numpy as np
import pandas as pd
import dagshub
from mlflow.models import infer_signature
from tqdm import tqdm
def get_data_loaders(batch_size=64, num_workers=4):
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
    
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()

    # Example usage: iterate through the training data
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break  # Just to demonstrate the output for one batch
    model= ConvolutionalNetwork()
    print(model)
    with open("model_summary.txt", "w") as f:
        f.write(str(model))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dagshub.init(repo_owner='mayank61', repo_name='mlfow_CNN_ImageClassification', mlflow=True)
    input_data = torch.randn(1, 1, 28, 28)  # Example input for signature inference
    output_data = model(input_data)  # Forward pass to get output
    signature = infer_signature(input_data, output_data)
    mlflow.set_experiment('MNIST_Convolutional_Network')
    with mlflow.start_run():
        mlflow.log_param('batch_size', 64)
        mlflow.log_param('learning_rate', 0.001)
        mlflow.log_param('total_parameters', total_parameters)
        mlflow.log_param('model_architecture', 'ConvolutionalNetwork')
        best_val_loss = float('inf')
        mlflow.log_artifact('model_summary.txt')
        mlflow.pytorch.log_model(model, "model", signature=signature)
        for epoch in tqdm(range(10)):
            training_loss = 0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                training_loss+=loss.item()
            training_loss /= len(train_loader)
            mlflow.log_metric('training_loss', training_loss,step=epoch)
            with torch.no_grad():
                test_loss = 0
                correct = 0
                for images, labels in test_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    pred = outputs.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                test_loss /= len(test_loader)
                if test_loss < best_val_loss:
                    best_val_loss = test_loss
                    torch.save(model.state_dict(), 'best_model.pth')
                accuracy = correct / len(test_loader.dataset)
                mlflow.log_metric('test_loss', test_loss, step=epoch)
                mlflow.log_metric('accuracy', accuracy, step=epoch)
        
        