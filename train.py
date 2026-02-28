"""
MNIST Handwritten Digit Classification using CNN
AI 100 - Midterm Project
Author: Paras Rathi

This script trains a Convolutional Neural Network (CNN) on the MNIST dataset
to classify handwritten digits (0-9).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
import os

# ─────────────────────────────────────────────
# 1. Hyperparameters
# ─────────────────────────────────────────────
BATCH_SIZE   = 64
LEARNING_RATE = 0.001
NUM_EPOCHS   = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2. Data Loading & Preprocessing
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),                        # convert to [0, 1] tensor
    transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean/std
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")
print(f"Classes          : {train_dataset.classes}")

# ─────────────────────────────────────────────
# 3. Model Architecture
# ─────────────────────────────────────────────
class CNN(nn.Module):
    """
    Simple two-block CNN:
      Block 1: Conv(1->32, 3x3) -> ReLU -> MaxPool(2x2)
      Block 2: Conv(32->64, 3x3) -> ReLU -> MaxPool(2x2)
      Classifier: FC(1600->128) -> ReLU -> Dropout(0.5) -> FC(128->10)
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN().to(DEVICE)
print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 4. Loss & Optimizer
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ─────────────────────────────────────────────
# 5. Training Loop
# ─────────────────────────────────────────────
train_losses, train_accs = [], []
test_losses,  test_accs  = [], []

def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

print("\n--- Training ---")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    tr_loss = running_loss / total
    tr_acc  = correct / total
    te_loss, te_acc = evaluate(test_loader)

    train_losses.append(tr_loss);  train_accs.append(tr_acc)
    test_losses.append(te_loss);   test_accs.append(te_acc)

    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
          f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc*100:.2f}%  |  "
          f"Test Loss: {te_loss:.4f}  Test Acc: {te_acc*100:.2f}%")

# ─────────────────────────────────────────────
# 6. Save Results to JSON
# ─────────────────────────────────────────────
results = {
    "train_losses": train_losses,
    "train_accs":   train_accs,
    "test_losses":  test_losses,
    "test_accs":    test_accs,
    "final_train_acc": train_accs[-1],
    "final_test_acc":  test_accs[-1],
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# ─────────────────────────────────────────────
# 7. Confusion Matrix
# ─────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))

# ─────────────────────────────────────────────
# 8. Plot Training Curves
# ─────────────────────────────────────────────
epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss')
axes[0].plot(epochs, test_losses,  'r-o', label='Test Loss')
axes[0].set_title('Loss per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, [a*100 for a in train_accs], 'b-o', label='Train Accuracy')
axes[1].plot(epochs, [a*100 for a in test_accs],  'r-o', label='Test Accuracy')
axes[1].set_title('Accuracy per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 9. Plot Confusion Matrix
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set(xticks=np.arange(10), yticks=np.arange(10),
       xticklabels=range(10), yticklabels=range(10),
       xlabel='Predicted Label', ylabel='True Label',
       title='Confusion Matrix - MNIST Test Set')
thresh = cm.max() / 2.0
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black', fontsize=7)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 10. Save Sample Predictions
# ─────────────────────────────────────────────
sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images[:16]
sample_labels = sample_labels[:16]
model.eval()
with torch.no_grad():
    sample_preds = torch.max(model(sample_images.to(DEVICE)), 1)[1].cpu()

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for idx, ax in enumerate(axes.flat):
    img = sample_images[idx].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    color = 'green' if sample_preds[idx] == sample_labels[idx] else 'red'
    ax.set_title(f"Pred:{sample_preds[idx].item()} True:{sample_labels[idx].item()}",
                 fontsize=8, color=color)
    ax.axis('off')
plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=10)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150)
plt.close()

print("\nAll outputs saved: results.json, training_curves.png, confusion_matrix.png, sample_predictions.png")

# Save model weights
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model weights saved to mnist_cnn.pth")
