from model import load_and_process_data
from model import BoxingMatchPredictor
import torch


X_new, _ = load_and_process_data('data_src/new_data.csv')


def load_model(input_size, num_classes, model_path = "best_model.pth"):
   model = BoxingMatchPredictor(input_size, num_classes)
   model.load_state_dict(torch.load(model_path))
   model.eval()  # Set the model to evaluation mode
   return model


input_size = X_new.shape[1]
num_classes = 3  # Should be the same as during training
model = load_model(input_size, num_classes, model_path='best_model.pth')

X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

with torch.no_grad():
   outputs = model(X_new_tensor)
   _, predicted = torch.max(outputs, 1)

for idx, pred in enumerate(predicted):
   outcome = pred.item()
   if outcome == 0:
      result = "First boxer wins"
   elif outcome == 1:
      result = "Second boxer wins"
   else:
      result = "Draw"
