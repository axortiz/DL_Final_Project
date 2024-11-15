import torch 
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


class BoxingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


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