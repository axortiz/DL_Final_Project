import argparse
from data_src import load_and_process_data
from model_src import BoxingMatchPredictor
import torch
import numpy as np

# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on boxing match data.')
    parser.add_argument('--model_type', type=str, choices=['kfold', 'mlp'], required=True,
                        help='Type of model to use for inference: "kfold" or "mlp".')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model file. Defaults to "best_model_kfold.pth" for kfold or "best_model.pth" for mlp.')
    parser.add_argument('--data_path', type=str, default='data_src/inference_data.csv',
                        help='Path to the inference data CSV file.')
    return parser.parse_args()

args = parse_arguments()

# Determine the model path based on model type if not provided
if args.model_path is None:
    if args.model_type == 'kfold':
        model_path = 'best_model_kfold.pth'
    else:
        model_path = 'best_model.pth'
else:
    model_path = args.model_path

# Load and process data
X_new, _, boxer_names = load_and_process_data(args.data_path, preserve_names=True)

def load_model(input_size, num_classes, model_path):
    model = BoxingMatchPredictor(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode
    return model

input_size = X_new.shape[1]
num_classes = 3  # Should be the same as during training

model = load_model(input_size, num_classes, model_path=model_path)
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
