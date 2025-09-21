import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_cifar10_dataloaders
from models.ViT import ViT
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
img_size = 32
num_epochs = 20
learning_rate = 3e-4
weight_decay = 0.01
save_path = "best_vit_cifar10.pth"

train_loader, val_loader = get_cifar10_dataloaders(batch_size=batch_size, img_size=img_size)

model = ViT(
    image_size=img_size,
    patch_size=4,
    num_classes=10,
    emb_dims=128,
    depth=4,
    num_heads=4,
    dropout=0.1,
    mlp_ratio=2
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# Training loop
best_val_acc = 0.0

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch [{epoch}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
    
    # save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved with Val Acc: {best_val_acc:.4f}")

print("Training finished. Best model saved at:", save_path)
