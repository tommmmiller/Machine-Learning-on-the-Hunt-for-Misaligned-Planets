import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, average_precision_score
)
import pandas as pd
import matplotlib.pyplot as plt

# === Load data ===
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
incl_test = np.load("incl_test.npy")  # contains inclination of each test example

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# === Define model (same as training) ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 75 * 75, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("simple_cnn.pt", map_location=device))
model.eval()

# === Predict ===
with torch.no_grad():
    logits = model(X_test_tensor.to(device))
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)

# === Compute metrics ===
acc = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auc_pr = average_precision_score(y_test, probs)

print(f"Accuracy:           {acc:.3f}")
print(f"Balanced Accuracy:  {bacc:.3f}")
print(f"F1 Score:           {f1:.3f}")
print(f"MCC:                {mcc:.3f}")
print(f"AUC-PR:             {auc_pr:.3f}")

# === Plot prediction correctness vs inclination ===
correct = (y_pred == y_test)

plt.figure(figsize=(8, 5))
plt.scatter(incl_test, correct, c=correct, cmap="bwr", alpha=0.7, label="Correct")
plt.axhline(0.5, color='gray', linestyle='--')
plt.xlabel("Inclination (deg)")
plt.ylabel("Prediction Correctness")
plt.title("Prediction Correctness vs Inclination")
plt.grid(True)
plt.tight_layout()
plt.savefig("inclination_vs_correctness.png")
print("Saved plot: inclination_vs_correctness.png")

np.save("y_pred.npy", y_pred)
np.save("y_prob.npy", probs)
