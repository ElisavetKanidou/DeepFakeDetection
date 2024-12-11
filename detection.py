import os
import numpy as np
import pandas as pd
import random
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms


# Step 1: Dataset και DataLoader
class ImageDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for file_name in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file_name)
            if file_name.endswith(('.jpg', '.png', '.jpeg')):
                self.image_paths.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Step 3: Paths to Test/Real and Test/Fake
test_dir = "/kaggle/input/deepfake-and-real-images/Dataset/Test"
real_dir = os.path.join(test_dir, "Real")
fake_dir = os.path.join(test_dir, "Fake")

# Verify the existence of directories
if not os.path.exists(real_dir):
    raise FileNotFoundError(f"Real directory not found: {real_dir}")
if not os.path.exists(fake_dir):
    raise FileNotFoundError(f"Fake directory not found: {fake_dir}")

# Count the number of images in each directory
num_real_images = len([f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
num_fake_images = len([f for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

# Print information
print(f"Found {num_real_images} Real images in directory: {real_dir}")
print(f"Found {num_fake_images} Fake images in directory: {fake_dir}")

# Dataset
real_dataset = ImageDataset(real_dir, 0, transform)
fake_dataset = ImageDataset(fake_dir, 1, transform)

dataset = real_dataset + fake_dataset
labels = real_dataset.labels + fake_dataset.labels

# Step 2: Stratified 5-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []


# Step 4: CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = None  # Θα οριστεί δυναμικά
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)  # Δυναμικός ορισμός
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training and Validation
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, labels)):
    print(f"Fold {fold + 1}")
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    # Split train into train and validation (5% for validation)
    val_size = int(0.05 * len(train_subset))
    train_size = len(train_subset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_subset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    # Model, Loss, Optimizer
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early Stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(50):  # Max epochs
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_fold{fold + 1}.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Step 5: Evaluate on Test Set
    model.load_state_dict(torch.load(f"best_model_fold{fold + 1}.pth"))
    model.eval()

    preds, actuals = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            preds.extend((outputs > 0.5).long().cpu().numpy())
            actuals.extend(labels.numpy())

    # Metrics
    report = classification_report(actuals, preds, output_dict=True)
    confusion = confusion_matrix(actuals, preds)

    # Save Metrics
    results.append({
        "Classifier": "SimpleCNN",
        "Set": "Test",
        "Fold": fold + 1,
        "Accuracy": report["accuracy"],
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-score": report["weighted avg"]["f1-score"]
    })

    # Confusion Matrix Plot
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])
    plt.title(f"Confusion Matrix - Fold {fold + 1}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Save Results to DataFrame
df_results = pd.DataFrame(results)
print(df_results)
