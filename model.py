import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import torch.nn as nn 
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

def load_and_process_data(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
    # Replace 'null' strings with NaN
    df.replace('null', np.nan, inplace=True)

    # Encode target variable with fixed classes
    def encode_winner(row):
        if row['f_boxer_result'] == "won":
            return 0  # First boxer wins
        elif row['f_boxer_result'] == "lost":
            return 1  # Second boxer wins
        else:
            return 2  # Draw

    df['winner_encoded'] = df.apply(encode_winner, axis=1)
    
    # Convert numeric columns to appropriate data types
    numeric_columns = [
        'f_boxer_age', 'f_boxer_height', 'f_boxer_reach',
        'f_boxer_won', 'f_boxer_lost', 'f_boxer_KOs',
        's_boxer_age', 's_boxer_height', 's_boxer_reach',
        's_boxer_won', 's_boxer_lost', 's_boxer_KOs',
        'matchRounds'
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['f_boxer_result', 'fightEnd']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    

    # Drop boxers' names
    df = df.drop(['f_boxer', 's_boxer'], axis=1)

    
    # Features and target
    X = df.drop(['winner', 'winner_encoded'], axis=1).values
    y = df['winner_encoded'].values
    
    return X, y

# Function to create the model

class BoxingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BoxingMatchPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BoxingMatchPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)


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
