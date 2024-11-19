from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from data_src import load_and_process_data, BoxingDataset
from model_src import BoxingMatchPredictor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


X, y = load_and_process_data('data_src/combined_data.csv')

NUM_FOLDS  = 5 

kfold = KFold(5, shuffle = True, random_state=42)

fold_accuracies = []

best_val_acc = 0.0

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
   X_train_fold, X_val_fold = X[train_idx], X[val_idx]
   y_train_fold, y_val_fold = y[train_idx], y[val_idx]

   train_dataset = BoxingDataset(X_train_fold, y_train_fold)
   val_dataset = BoxingDataset(X_val_fold, y_val_fold)

   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32)

   model = BoxingMatchPredictor(X_train_fold.shape[1], 3)
   optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
   criterion = nn.CrossEntropyLoss()

   for epoch in range(50):
      model.train() 

      for X_batch, y_batch in train_loader:
         optimizer.zero_grad()
         outputs = model(X_batch)
         loss = criterion(outputs, y_batch)
         loss.backward()
         optimizer.step()

      
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

