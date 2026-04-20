import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath="data/traffic.csv"):
    df = pd.read_csv(filepath)
    
    # Auto-detect Date column
    date_cols = [c for c in df.columns if c.lower() in ['date', 'datetime', 'time', 'date_time', 'timestamp']]
    date_col = date_cols[0] if date_cols else df.select_dtypes(include=['object', 'datetime']).columns[0]
    
    # Auto-detect Traffic Volume column
    volume_cols = [c for c in df.columns if any(word in c.lower() for word in ['volume', 'count', 'traffic', 'vehicles'])]
    volume_col = volume_cols[0] if volume_cols else df.select_dtypes(include=['number']).columns[0]
    
    # Auto-detect Area Name column
    area_cols = [c for c in df.columns if any(word in c.lower() for word in ['area', 'location', 'city', 'intersection', 'road'])]
    area_col = area_cols[0] if area_cols else None

    # Rename columns to standard format
    rename_mapping = {date_col: "datetime", volume_col: "traffic_volume"}
    if area_col:
        rename_mapping[area_col] = "area_name"
    df = df.rename(columns=rename_mapping)
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime', 'traffic_volume']).copy()
    
    # Sort dataset by datetime
    df = df.sort_values(by='datetime').reset_index(drop=True)
    
    # Feature Engineering
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day'] >= 5).astype(int)
    
    return df

def prepare_rf_data(df):
    """Creates lag features for Random Forest model."""
    df_rf = df.copy()
    df_rf['lag_1'] = df_rf['traffic_volume'].shift(1)
    df_rf['lag_2'] = df_rf['traffic_volume'].shift(2)
    df_rf['lag_3'] = df_rf['traffic_volume'].shift(3)
    
    # Drop resulting NaN rows from lag creation
    df_rf = df_rf.dropna().reset_index(drop=True)
    
    # Features to train on
    features = ['hour', 'day', 'is_weekend', 'lag_1', 'lag_2', 'lag_3']
    target = 'traffic_volume'
    
    return df_rf[features], df_rf[target]

def create_lstm_sequences(data, seq_length=24):
    """Creates sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def get_traffic_category(predicted_volume, df):
    """
    Categorizes the traffic volume into Low, Medium, or High based on the
    historical percentiles from the original dataset.
    """
    p33 = df['traffic_volume'].quantile(0.33)
    p66 = df['traffic_volume'].quantile(0.66)
    
    if predicted_volume < p33:
        return 'Low'
    elif predicted_volume < p66:
        return 'Medium'
    else:
        return 'High'
