import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score
from data_src import load_and_process_data


X_full, y_full = load_and_process_data('data_src/combined_data.csv')


# Use StratifiedKFold to maintain class distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

# Variables to keep track of the best model
best_val_acc = 0.0
best_model = None

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
    X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
    y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
    
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': 0.1,
        'verbose': -1
    }
    
    callbacks = [
      early_stopping(stopping_rounds=10),
      log_evaluation(10)  # Adjust the logging frequency as needed
   ]
    
    gbm = lgb.train(
      params,
      train_data,
      num_boost_round=100,
      valid_sets=[train_data, val_data],
      callbacks=callbacks)
    
    # Predict
    y_pred = gbm.predict(X_val_fold, num_iteration=gbm.best_iteration)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    acc = accuracy_score(y_val_fold, y_pred_classes)
    print(f"Fold {fold+1} Validation Accuracy: {acc:.4f}")
    fold_accuracies.append(acc)
    
    # Check if this is the best model so far
    if acc > best_val_acc:
        best_val_acc = acc
        best_model = gbm
        print(f"New best model found at Fold {fold+1} with accuracy {acc:.4f}")

# Save the best model
if best_model is not None:
    best_model.save_model('best_model_lgb.pth')
    print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")
else:
    print("No model was saved.")

mean_accuracy = np.mean(fold_accuracies)
print(f"Mean Cross-Validation Accuracy: {mean_accuracy:.4f}")
