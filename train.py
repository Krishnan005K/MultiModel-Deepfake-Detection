import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RealFakeAudioDataset
from model import FakeAudioDetector
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = RealFakeAudioDataset("data/train/real", "data/train/fake")
val_ds   = RealFakeAudioDataset("data/val/real", "data/val/fake")

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=32)

model = FakeAudioDetector().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- store metrics for visualization ---
train_losses = []
val_accuracies = []

for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for X, y in train_dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dl)
    train_losses.append(avg_loss)

    # validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.to(device), y.to(device)
            out = model(X)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    val_accuracies.append(acc)

    print(f"Epoch {epoch}: Train Loss {avg_loss:.4f}, Val Acc {acc:.4f}")

# --- plot metrics ---
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(1, 21), train_losses, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(range(1, 21), val_accuracies, marker='o', color='orange')
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()

# Save model
torch.save(model.state_dict(), "fake_audio_detector.pth")
