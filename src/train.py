import os
import torch
import mlflow
import mlflow.pytorch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

# Create models folder
os.makedirs("models", exist_ok=True)

mlflow.set_experiment("mnist_baseline")

transform = transforms.ToTensor()


train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model = SimpleCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with mlflow.start_run():

    mlflow.log_param("batch_size", 64)
    mlflow.log_param("learning_rate", 0.001)

    for epoch in range(2):
        total_loss = 0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mlflow.log_metric("loss", total_loss, step=epoch)

    torch.save(model.state_dict(), "models/model.pt")
    mlflow.log_artifact("models/model.pt")

print("Training complete. Model saved.")