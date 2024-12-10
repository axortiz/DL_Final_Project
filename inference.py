from data_src import load_and_process_data
from model_src import BoxingMatchPredictor
import torch
import numpy as np



X_new, _, boxer_names = load_and_process_data('data_src/inference_data.csv', preserve_names=True)


def load_model(input_size, num_classes, model_path):
    model = BoxingMatchPredictor(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode
    return model


input_size = X_new.shape[1]
num_classes = 3  # Should be the same as during training

model = load_model(input_size, num_classes, model_path='best_model_kfold.pth')
# model = load_model(input_size, num_classes, model_path='best_model.pth')

X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

with torch.no_grad():
    outputs = model(X_new_tensor)
    _, predicted = torch.max(outputs, 1)

for idx, pred in enumerate(predicted):
    outcome = pred.item()
    boxer1, boxer2 = boxer_names[idx]
    if outcome == 0:
        result = f"{boxer1} wins"
    elif outcome == 1:
        result = f"{boxer2} wins"
    else:
        result = "Draw"
    print(f"Match {idx + 1}: {result}")
