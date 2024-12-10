from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from data_src import load_and_process_data, BoxingDataset
from model_src import BoxingMatchPredictor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


X, y = load_and_process_data('data_src/combined_data.csv')

NUM_FOLDS  = 5 

kfold = KFold(5, shuffle = True, random_state=42)

fold_accuracies = []

best_val_acc = 0.0

fold_accuracies = []
best_val_acc = 0.0
all_fold_train_losses = []
all_fold_val_losses = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
   X_train_fold, X_val_fold = X[train_idx], X[val_idx]
   y_train_fold, y_val_fold = y[train_idx], y[val_idx]

   train_dataset = BoxingDataset(X_train_fold, y_train_fold)
   val_dataset = BoxingDataset(X_val_fold, y_val_fold)

   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32)

   model = BoxingMatchPredictor(X_train_fold.shape[1], 3)
   optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay=0.01)
   criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

   train_losses = []
   val_losses = []

   for epoch in range(100):
      model.train()
      epoch_train_loss = 0.0
      num_train_batches = 0

      for X_batch, y_batch in train_loader:
         optimizer.zero_grad()
         outputs = model(X_batch)
         loss = criterion(outputs, y_batch)
         loss.backward()
         optimizer.step()
         epoch_train_loss += loss.item()
         num_train_batches += 1
      
      avg_train_loss = epoch_train_loss / num_train_batches
      train_losses.append(avg_train_loss)

      model.eval()
      epoch_val_loss = 0.0
      num_val_batches = 0
      
      with torch.no_grad():
         for X_val_batch, y_val_batch in val_loader:
            outputs = model(X_val_batch)
            val_loss = criterion(outputs, y_val_batch)
            epoch_val_loss += val_loss.item()
            num_val_batches += 1
      
      avg_val_loss = epoch_val_loss / num_val_batches
      val_losses.append(avg_val_loss)

      print(f"Fold {fold + 1}, Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

   all_fold_train_losses.append(train_losses)
   all_fold_val_losses.append(val_losses)

   model.eval()
   all_preds = []
   all_labels = []

   with torch.no_grad():
      for X_val_batch, y_val_batch in val_loader:
         outputs = model(X_val_batch)
         _, predicted = torch.max(outputs, 1)
         all_preds.extend(predicted.numpy())
         all_labels.extend(y_val_batch.numpy())
   
   fold_acc = accuracy_score(all_labels, all_preds)
   print(f"Fold {fold + 1} Validation accuracy: {fold_acc:.4f}")

   fold_accuracies.append(fold_acc)

   if fold_acc > best_val_acc:
      best_val_acc = fold_acc
      torch.save(model.state_dict(), 'best_model_kfold.pth')


mean_acc = sum(fold_accuracies) / NUM_FOLDS
print(f"Mean validation accuracy: {mean_acc:.4f}")

# Plot losses for each fold
plt.figure(figsize=(12, 8))
for fold in range(NUM_FOLDS):
    plt.subplot(2, 3, fold + 1)
    plt.plot(all_fold_train_losses[fold], label='Train Loss')
    plt.plot(all_fold_val_losses[fold], label='Val Loss')
    plt.title(f'Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

plt.tight_layout()
plt.savefig('loss_plots.png')
plt.show()
plt.close()

# Plot average losses across all folds
avg_train_losses = np.mean(all_fold_train_losses, axis=0)
avg_val_losses = np.mean(all_fold_val_losses, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(avg_train_losses, label='Avg Train Loss')
plt.plot(avg_val_losses, label='Avg Val Loss')
plt.title('Average Losses Across All Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.savefig('average_loss_plot.png')
plt.show()
plt.close()

