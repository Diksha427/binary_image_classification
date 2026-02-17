import os
import torch
import mlflow
import mlflow.pytorch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import SimpleCNN
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create models folder
os.makedirs("models", exist_ok=True)

# Set MLflow experiment
mlflow.set_experiment("mnist_baseline")

# Data transform
transform = transforms.ToTensor()

# Load dataset
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Model, loss, optimizer
model = SimpleCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", epochs)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Log average loss per epoch
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    # Save model locally
    torch.save(model.state_dict(), "models/model.pt")

    # Log full model to MLflow
    mlflow.pytorch.log_model(model, "model")

    # --------------------------
    # Evaluation + Confusion Matrix
    # --------------------------

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

print("Training complete. Model saved.")
