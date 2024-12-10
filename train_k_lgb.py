import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score
from data_src import load_and_process_data
import matplotlib.pyplot as plt

X_full, y_full = load_and_process_data('data_src/combined_data.csv')

# Use StratifiedKFold to maintain class distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
all_fold_train_losses = []
all_fold_val_losses = []

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
        'learning_rate': 0.05,
        'verbose': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 20,
    }
    
    # Create lists to store the metrics
    train_losses = []
    val_losses = []
    
    def callback_metrics(env):
        train_losses.append(env.evaluation_result_list[0][2])
        val_losses.append(env.evaluation_result_list[1][2])
    
    callbacks = [
        early_stopping(stopping_rounds=20),
        log_evaluation(10),
        callback_metrics
    ]
    
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        callbacks=callbacks
    )
    
    # Store losses for this fold
    all_fold_train_losses.append(train_losses)
    all_fold_val_losses.append(val_losses)
    
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
    best_model.save_model('best_model_lgb.txt')
    print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")

mean_accuracy = np.mean(fold_accuracies)
print(f"Mean Cross-Validation Accuracy: {mean_accuracy:.4f}")

# Plot losses for each fold
plt.figure(figsize=(15, 10))
for fold in range(len(all_fold_train_losses)):
    plt.subplot(2, 3, fold + 1)
    plt.plot(all_fold_train_losses[fold], label='Train Loss')
    plt.plot(all_fold_val_losses[fold], label='Val Loss')
    plt.title(f'Fold {fold + 1}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log loss)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('lgb_loss_plots.png')
plt.show()
plt.close()

# Plot average losses across all folds
avg_train_losses = np.mean(np.array(all_fold_train_losses), axis=0)
avg_val_losses = np.mean(np.array(all_fold_val_losses), axis=0)

plt.figure(figsize=(10, 6))
plt.plot(avg_train_losses, label='Avg Train Loss')
plt.plot(avg_val_losses, label='Avg Val Loss')
plt.title('LightGBM: Average Losses Across All Folds')
plt.xlabel('Iteration')
plt.ylabel('Loss (log loss)')
plt.legend()
plt.grid(True)
plt.savefig('lgb_average_loss_plot.png')
plt.show()
plt.close()

# Plot accuracy distribution
# plt.figure(figsize=(8, 6))
# plt.boxplot(fold_accuracies)
# plt.title('LightGBM: Validation Accuracy Distribution Across Folds')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.savefig('lgb_accuracy_distribution.png')
# plt.show()
# plt.close()
