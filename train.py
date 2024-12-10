import torch.nn as nn 
import torch
from model_src import BoxingMatchPredictor 
from data_src import load_and_process_data, BoxingDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
WEIGHT_DECAY = 0.05

# Load and normalize data
X_train, y_train = load_and_process_data('data_src/train.csv')
X_val, y_val = load_and_process_data('data_src/validation.csv')


# Normalize the data
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_train = (X_train - train_mean) / (train_std + 1e-7)
X_val = (X_val - train_mean) / (train_std + 1e-7)

train_dataset = BoxingDataset(X_train, y_train)
val_dataset = BoxingDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model
input_size = X_train.shape[1]
num_classes = 3
model = BoxingMatchPredictor(input_size, num_classes)

# Loss, optimizer and scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)
)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=True
)

# Training tracking
best_val_acc = 0.0
best_model_state = None

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        
        _, predicted = torch.max(outputs, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            outputs = model(X_val_batch)
            loss = criterion(outputs, y_val_batch)
            val_loss += loss.item() * X_val_batch.size(0)
            
            _, predicted = torch.max(outputs, 1)
            val_total += y_val_batch.size(0)
            val_correct += (predicted == y_val_batch).sum().item()

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    # Learning rate scheduling
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model.pth')
        print('New best model saved!')

# Plot losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_validation_loss_mlp.png')
plt.close()

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_validation_accuracy_mlp.png')
plt.close()

print(f"Best validation accuracy: {best_val_acc:.4f}")
