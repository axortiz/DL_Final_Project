import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Function to clean data


def clean_data(df):
    # Replace 'null', 'Null', 'not listed', 'Unknown' with NaN
    df = df.replace(['null', 'Null', 'not listed', 'Unknown',
                    'unknown', 'Not listed'], np.nan)

    # Convert numerical columns to numeric dtype
    num_cols = ['f_boxer_age', 'f_boxer_height', 'f_boxer_reach', 'f_boxer_won', 'f_boxer_lost', 'f_boxer_KOs',
                's_boxer_age', 's_boxer_height', 's_boxer_reach', 's_boxer_won', 's_boxer_lost', 's_boxer_KOs', 'matchRounds']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing numerical values with mean
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    # For categorical columns, fill missing values with 'Unknown'
    cat_cols = ['f_boxer', 's_boxer', 'f_boxer_result', 'fightEnd', 'winner']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    return df

# Function to preprocess data


def preprocess_data(df):
    # Clean data
    df = clean_data(df)

    # Encode 'winner' column: 1 if 'winner' == 'f_boxer', 0 if 'winner' == 's_boxer'
    df['winner_encoded'] = df.apply(lambda row: 1 if row['winner'] == row['f_boxer'] else (
        0 if row['winner'] == row['s_boxer'] else np.nan), axis=1)

    # Drop rows with NaN in 'winner_encoded'
    df = df.dropna(subset=['winner_encoded'])

    # Drop unnecessary columns
    df = df.drop(columns=['f_boxer', 's_boxer',
                 'f_boxer_result', 'fightEnd', 'winner'])

    # Separate features and target
    X = df.drop(columns=['winner_encoded'])
    y = df['winner_encoded']

    return X, y

# Function to create the model


def create_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Load datasets
train_df = pd.read_csv('Training.csv')
val_df = pd.read_csv('validation.csv')

# Preprocess training data
train_X, train_y = preprocess_data(train_df)

# Fit scaler on training data
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

# Preprocess validation data
val_X, val_y = preprocess_data(val_df)

# Use the same scaler to transform validation data
val_X_scaled = scaler.transform(val_X)

# Create the model
input_shape = train_X_scaled.shape[1]
model = create_model(input_shape)

# Train the model
history = model.fit(train_X_scaled, train_y, epochs=50,
                    batch_size=16, validation_data=(val_X_scaled, val_y))

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_X_scaled, val_y)
print(f"Validation Accuracy: {val_accuracy}")

# Make predictions
predictions = model.predict(val_X_scaled)
predicted_classes = (predictions > 0.5).astype("int32")

# Print the predicted and actual classes
print("Predicted classes:", predicted_classes.flatten())
print("Actual classes:", val_y.values)
