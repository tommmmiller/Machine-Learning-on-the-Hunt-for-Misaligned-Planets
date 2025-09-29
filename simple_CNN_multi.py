import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import os

# === Load Data ===
X_train = torch.load("X_train.pt").numpy()
y_train = torch.load("y_train_multi.pt").numpy()
X_test = torch.load("X_test.pt").numpy()
y_test = torch.load("y_test_multi.pt").numpy()

# === Convert to PyTorch tensors ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# === Define Simple CNN ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 150x150
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 75x75
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 75 * 75, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4-class classification
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# === Training setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Train ===
model.train()
for epoch in range(100):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# === Evaluate ===
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    y_pred = outputs.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["0-3°", "3-9°", "9-18°", "18-25°"]))

# === Save model ===
torch.save(model.state_dict(), "simple_cnn_multi.pt")
print("Model saved to simple_cnn_multi.pt")
