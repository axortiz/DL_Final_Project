import torch.nn as nn 
import torch
from model_src import BoxingMatchPredictor 
from data_src import load_and_process_data, BoxingDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


X_train, y_train  = load_and_process_data('data_src/train.csv')
X_val, y_val = load_and_process_data('data_src/validation.csv')

train_dataset = BoxingDataset(X_train, y_train)
val_dataset = BoxingDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model
input_size = X_train.shape[1]
num_classes = 3  # Fixed number of classes
model = BoxingMatchPredictor(input_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_acc = 0.0
best_model_state = None


train_losses = []
val_losses = []

num_epochs = 50
# Training loop 
for epoch in range(num_epochs):
    model.train() 

    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)


    # Validation 
    model.eval() 
    val_running_loss = 0.0 
    correct = 0 
    total = 0 

    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            outputs = model(X_val_batch)
            val_loss = criterion(outputs, y_val_batch)
            val_running_loss += val_loss.item() * X_val_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_val_batch.size(0)
            correct += (predicted == y_val_batch).sum().item()
        
    val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_acc = correct / total

    
    print(f"Number of correct predictions: {correct}, Total predictions: {total}")
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model.pth')
        print('Best model saved')

    
# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
