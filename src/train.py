import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import only SimpleCNN (since that's what you're using)
from src.model import SimpleCNN

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================
# CONFIGURATION - YOU CAN CHANGE THESE VALUES
# ============================================
CONFIG = {
    "model_type": "simple_cnn",     # Only using simple_cnn
    "batch_size": 32,                # You can change this (16, 32, 64, etc.)
    "learning_rate": 0.001,          # You can change this (0.01, 0.001, 0.0001)
    "epochs": 10,                     # You can change this (5, 10, 20, etc.)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "experiment_name": "cats_vs_dogs_simple_cnn",
    "input_size": (224, 224),
    "num_classes": 2,
    "num_workers": 2,                 # For data loading
    "seed": 42                         # For reproducibility
}

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def get_transforms():
    """Get data transforms with augmentation for training"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_data():
    """Load and prepare datasets"""
    print("\n=== Loading Data ===")
    
    train_transform, val_transform = get_transforms()
    
    # Check if data exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}. Run data.py first.")
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=str(DATA_DIR / "train"),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=str(DATA_DIR / "val"),
        transform=val_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=str(DATA_DIR / "test"),
        transform=val_transform
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if CONFIG["device"] == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating'):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def plot_confusion_matrix(cm, classes, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """Main training function"""
    print("\n" + "="*60)
    print("CATS VS DOGS CLASSIFIER TRAINING")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(CONFIG["seed"])
    
    # Print configuration
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Set device
    device = torch.device(CONFIG["device"])
    print(f"\nUsing device: {device}")
    
    # Set up MLflow
    mlflow.set_experiment(CONFIG["experiment_name"])
    
    # Load data
    train_loader, val_loader, test_loader, classes = load_data()
    
    # Initialize model
    print(f"\nInitializing SimpleCNN model...")
    model = SimpleCNN(num_classes=CONFIG["num_classes"])
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    # Learning rate scheduler - FIXED: removed verbose parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(CONFIG)
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        for epoch in range(1, CONFIG["epochs"] + 1):
            print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Print progress
            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODELS_DIR / 'best_model.pt')
                print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        # Test the model
        print("\nEvaluating on test set...")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        test_acc = accuracy_score(all_labels, all_preds) * 100
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
        
        print(f"\nTest Accuracy: {test_acc:.2f}%")
        
        # Log test metrics
        mlflow.log_metrics({
            "test_accuracy": test_acc
        })
        
        # Plot and save confusion matrix
        cm_path = MODELS_DIR / "confusion_matrix.png"
        plot_confusion_matrix(cm, classes, cm_path)
        mlflow.log_artifact(str(cm_path))
        
        # Plot training curves
        curves_path = MODELS_DIR / "training_curves.png"
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)
        mlflow.log_artifact(str(curves_path))
        
        # Save classification report
        report_path = MODELS_DIR / "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))
        
        # Save final model
        final_model_path = MODELS_DIR / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'classes': classes,
            'test_accuracy': test_acc
        }, final_model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")
        
        print(f"\n Training complete!")
        print(f"Model saved to: {final_model_path}")
        print(f"MLflow Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()