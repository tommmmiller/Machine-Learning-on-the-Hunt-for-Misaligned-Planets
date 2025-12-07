import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import os

# === Paths / config ===
BASE_DIR = "."  # change if your .pt training files are elsewhere

num_epochs = 50          # max epochs
batch_size = 16
early_stopping = True
patience = 5             # epochs with no improvement before stopping

# === Load Data from .pt ===
X_train = torch.load(os.path.join(BASE_DIR, "X_train.pt"))
y_train = torch.load(os.path.join(BASE_DIR, "y_train.pt"))
X_val   = torch.load(os.path.join(BASE_DIR, "X_val.pt"))
y_val   = torch.load(os.path.join(BASE_DIR, "y_val.pt"))
X_test  = torch.load(os.path.join(BASE_DIR, "X_test.pt"))
y_test  = torch.load(os.path.join(BASE_DIR, "y_test.pt"))

# If these are numpy arrays, this keeps behaviour consistent
if isinstance(X_train, np.ndarray):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
else:
    X_train_tensor = X_train.to(torch.float32)
    X_val_tensor   = X_val.to(torch.float32)
    X_test_tensor  = X_test.to(torch.float32)

if isinstance(y_train, np.ndarray):
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)
else:
    y_train_tensor = y_train.to(torch.long)
    y_val_tensor   = y_val.to(torch.long)
    y_test_tensor  = y_test.to(torch.long)

# === Datasets & loaders ===
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

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

best_val_loss = float("inf")
best_model_state = None
patience_counter = 0

# === Train ===
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0

    # --- Training loop ---
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)

    # --- Validation loop ---
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            all_val_preds.append(preds.cpu())
            all_val_targets.append(targets.cpu())

    val_loss /= len(val_loader.dataset)
    all_val_preds = torch.cat(all_val_preds).numpy()
    all_val_targets = torch.cat(all_val_targets).numpy()
    val_acc = accuracy_score(all_val_targets, all_val_preds)

    print(f"Epoch {epoch+1}: "
          f"Train Loss = {epoch_train_loss:.4f} | "
          f"Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.3f}")

    # --- Early stopping ---
    if early_stopping:
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.train()

# Load best model if early stopping was used
if early_stopping and best_model_state is not None:
    model.load_state_dict(best_model_state)
    model.to(device)

# === Evaluate on test set ===
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.to(device)).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test_tensor.numpy(), y_pred)
    print(f"\nTest Accuracy: {acc:.3f}")

# === Save model ===
torch.save(model.state_dict(), "simple_cnn.pt")
print("Model saved to simple_cnn.pt")
