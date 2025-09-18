import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import os

# === Load Data ===
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

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
            nn.Linear(64, 2)  # binary classification
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
for epoch in range(10):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}")

# === Evaluate ===
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.to(device)).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.3f}")

# === Save model ===
torch.save(model.state_dict(), "simple_cnn.pt")
print("Model saved to simple_cnn.pt")
