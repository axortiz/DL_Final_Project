import lightgbm as lgb
from data_src import load_and_process_data
import numpy as np
import pandas as pd

# Load the model
best_model = lgb.Booster(model_file='best_model_lgb.txt')  # or 'best_model_lgb.pth' if you saved it with that extension

# Load and preprocess new data

X_new, boxers_df = load_and_process_data('data_src/inference_data.csv')

# Make predictions
y_pred = best_model.predict(X_new)
y_pred_classes = np.argmax(y_pred, axis=1)

# Interpret and display predictions
for idx, pred in enumerate(y_pred_classes):
    outcome = pred
    if outcome == 0:
        result = "First boxer wins"
    elif outcome == 1:
        result = "Second boxer wins"
    else:
        result = "Draw"
    
    f_boxer_name = boxers_df.iloc[idx]['f_boxer']
    s_boxer_name = boxers_df.iloc[idx]['s_boxer']
    
    print(f"Match {idx+1}: {f_boxer_name} vs {s_boxer_name} - Predicted Outcome: {result}")
