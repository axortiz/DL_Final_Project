import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score
from data_src import load_and_process_data
import matplotlib.pyplot as plt

# Load data
X_full, y_full = load_and_process_data('data_src/train.csv')

# Calculate class distribution
unique, counts = np.unique(y_full, return_counts=True)
total_samples = len(y_full)

# Calculate weights to penalize draws more heavily
weights = np.ones(len(y_full))
for idx, label in enumerate(y_full):
    if label == 2:  # If it's a draw
        weights[idx] = 0.5  # Reduce the weight of draws
    else:
        weights[idx] = 1.0  # Keep normal weight for wins

# Use StratifiedKFold to maintain class distribution
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_accuracies = []
all_fold_train_losses = []
all_fold_val_losses = []

# Variables to keep track of the best model
best_val_acc = 0.0
best_model = None

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
    X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
    y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
    weights_train = weights[train_idx]
    
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold, weight=weights_train)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
    # Modified parameters to discourage draws
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,  # Increased learning rate
        'verbose': -1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'num_leaves': 63,  # Increased complexity
        'max_depth': 8,  # Increased depth
        'min_data_in_leaf': 10,  # Reduced to allow more specific rules
        'boost_from_average': True,
        'reg_alpha': 0.05,  # Reduced regularization
        'reg_lambda': 0.05,  # Reduced regularization
        'is_unbalance': False,  # We're handling balance through weights
        'min_gain_to_split': 0.5,  # Increased to encourage more decisive splits
        'min_sum_hessian_in_leaf': 1e-3,  # Reduced to allow more specific rules
    }
    
    # Create lists to store the metrics
    train_losses = []
    val_losses = []
    
    def callback_metrics(env):
        train_losses.append(env.evaluation_result_list[0][2])
        val_losses.append(env.evaluation_result_list[1][2])
    
    callbacks = [
        early_stopping(stopping_rounds=30),
        log_evaluation(10),
        callback_metrics
    ]
    
    # Train the model
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=callbacks
    )
    
    # Store losses for this fold
    all_fold_train_losses.append(train_losses)
    all_fold_val_losses.append(val_losses)
    
    # After the loop, ensure all lists are of the same length
    max_length = max(len(loss) for loss in all_fold_train_losses)
    for i in range(len(all_fold_train_losses)):
        while len(all_fold_train_losses[i]) < max_length:
            all_fold_train_losses[i].append(np.nan)  # Pad with NaN
    while len(all_fold_val_losses[i]) < max_length:
        all_fold_val_losses[i].append(np.nan)  # Pad with NaN
    
    # Predict
    y_pred = gbm.predict(X_val_fold, num_iteration=gbm.best_iteration)
    
    # Add a small penalty for draws in prediction
    y_pred[:, 2] *= 0.8  # Reduce the probability of draws
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

# Plot training curves
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
plt.close()

# Plot average losses
avg_train_losses = np.nanmean(np.array(all_fold_train_losses), axis=0)
avg_val_losses = np.nanmean(np.array(all_fold_val_losses), axis=0)

plt.figure(figsize=(10, 6))
plt.plot(avg_train_losses, label='Avg Train Loss')
plt.plot(avg_val_losses, label='Avg Val Loss')
plt.title('LightGBM: Average Losses Across All Folds')
plt.xlabel('Iteration')
plt.ylabel('Loss (log loss)')
plt.legend()
plt.grid(True)
plt.savefig('lgb_average_loss_plot.png')
plt.close()
