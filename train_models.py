import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from utils import load_and_preprocess_data, prepare_rf_data, create_lstm_sequences

def train_random_forest(df):
    print("Training Random Forest model...")
    X, y = prepare_rf_data(df)
    
    # Train-test split (80-20, no shuffling)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train RF model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    preds = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"Random Forest RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    # Save model
    joblib.dump(rf_model, 'models/random_forest.pkl')
    print("Saved Random Forest model to models/random_forest.pkl")
    return rmse, mae

def train_lstm(df):
    print("Training LSTM model...")
    data = df[['traffic_volume']].values
    
    # Train-test split (80-20, no shuffling)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Normalize data
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    # Create sequences
    seq_length = 24
    if len(train_data_scaled) <= seq_length or len(test_data_scaled) <= seq_length:
        print("Not enough data to create sequences. Sequence length:", seq_length)
        return
        
    X_train, y_train = create_lstm_sequences(train_data_scaled, seq_length)
    X_test, y_test = create_lstm_sequences(test_data_scaled, seq_length)
    
    # Reshape for LSTM (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(seq_length, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Evaluate
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    y_test_inv = scaler.inverse_transform(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_inv, preds))
    mae = mean_absolute_error(y_test_inv, preds)
    print(f"LSTM RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    # Save model and scaler
    model.save('models/lstm.keras')
    joblib.dump(scaler, 'models/scaler_y.pkl')
    print("Saved LSTM model and scaler")
    return rmse, mae

if __name__ == "__main__":
    df = load_and_preprocess_data()
    rf_rmse, rf_mae = train_random_forest(df)
    lstm_rmse, lstm_mae = train_lstm(df)
    
    metrics = {
        "rf": {"rmse": float(rf_rmse), "mae": float(rf_mae)},
        "lstm": {"rmse": float(lstm_rmse), "mae": float(lstm_mae)}
    }
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Saved models/metrics.json")
