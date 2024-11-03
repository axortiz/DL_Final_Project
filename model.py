import tensorflow as tf
from tensorflow import Sequential, Dense, Dropout, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

# Load your dataset
# Assume 'df' is a pandas DataFrame containing your dataset
# df = pd.read_csv('your_data.csv')

# Preprocess the data


def preprocess_data(df):
    # Drop columns you won't use (e.g., f_boxer, s_boxer, and any non-numeric columns)
    # Assuming 'f_boxer', 's_boxer', and 'f_boxer_result' columns are for the boxers' names and result, which we don't need numerically
    df = df.drop(columns=['f_boxer', 's_boxer'])

    # Encode categorical labels (e.g., fight outcome or winner) if needed
    # Assuming 'winner' is the target column
    label_encoder = LabelEncoder()
    df['winner'] = label_encoder.fit_transform(df['winner'])

    # Split into features (X) and target (y)
    X = df.drop(columns=['winner'])
    y = df['winner']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Assuming 'df' is your dataset
# X, y = preprocess_data(df)


# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Build the model


def create_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        # Use sigmoid for binary classification (winner prediction)
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Create and train the model
input_shape = X_train.shape[1]
model = create_model(input_shape)

# Train the model
history = model.fit(X_train, y_train, epochs=50,
                    batch_size=16, validation_data=(X_val, y_val))

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy}")

# Example of making predictions
predictions = model.predict(X_val)
predicted_classes = (predictions > 0.5).astype("int32")
