import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from tensorflow.keras.models import load_model
from utils import load_and_preprocess_data, get_traffic_category, create_lstm_sequences

# Page Configuration
st.set_page_config(
    page_title="Traffic Volume Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Refresh Data & Models"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #2c3e50; }
        .winner-text { color: #27ae60; font-weight: bold; }
        .badge-low { background-color: #2ecc71; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        .badge-medium { background-color: #f39c12; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        .badge-high { background-color: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        return load_and_preprocess_data()
    except Exception as e:
        st.error(f"Error loading data: {e}. Make sure data/traffic.csv exists.")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('models/random_forest.pkl')
        lstm_model = load_model('models/lstm.keras', compile=False)
        scaler = joblib.load('models/scaler_y.pkl')
        
        metrics = None
        try:
            with open('models/metrics.json', 'r') as f:
                metrics = json.load(f)
        except Exception:
            pass
            
        return rf_model, lstm_model, scaler, metrics
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run train_models.py first.")
        return None, None, None, None

df = load_data()
rf_model, lstm_model, scaler, metrics = load_models()

if df.empty:
    st.stop()

# --- HEADER & AREA FILTER ---
selected_area = "All Areas"
if 'area_name' in df.columns:
    areas = df['area_name'].dropna().unique().tolist()
    if len(areas) > 0:
        with st.sidebar:
            st.divider()
            st.header("Filters")
            selected_area = st.selectbox("Select Area", ["All Areas"] + areas)
        
        if selected_area != "All Areas":
            df = df[df['area_name'] == selected_area].reset_index(drop=True)

if selected_area == "All Areas":
    # Try to guess a "main area" from the original dataframe's first element or just generic
    main_area = df['area_name'].iloc[0] if 'area_name' in df.columns and len(df) > 0 else "All Areas"
    st.title(f"🚦 Traffic Volume Dashboard - ({main_area} Data)")
else:
    st.title(f"🚦 Traffic Volume Dashboard - {selected_area}")

st.markdown("Analyze historical traffic patterns and predict future volume using **Random Forest** and **Deep Learning (LSTM)** models.")
st.divider()

# --- TOP SECTION ---
col1, col2 = st.columns([1, 2], gap="large")
with col1:
    st.subheader("Data Preview")
    # Show clean formatted date
    preview_df = df.copy()
    if 'datetime' in preview_df.columns:
        preview_df['datetime'] = preview_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(preview_df.head(), use_container_width=True)

with col2:
    st.subheader("Traffic Trend Validation")
    
    # Plot last N points for clarity
    plot_points = min(400, len(df))
    plot_df = df.tail(plot_points).copy()
    
    if 'area_name' in plot_df.columns and selected_area == "All Areas":
        fig = px.line(plot_df, x='datetime', y='traffic_volume', color='area_name', 
                      title="Traffic Volume (All Areas)",
                      labels={'datetime': 'Date Time', 'traffic_volume': 'Traffic Volume', 'area_name': 'Area'})
    else:
        fig = px.line(plot_df, x='datetime', y='traffic_volume',
                      title=f"Traffic Volume ({selected_area})",
                      labels={'datetime': 'Date Time', 'traffic_volume': 'Traffic Volume'})
    
    fig.update_layout(
        xaxis_title="Date Time",
        yaxis_title="Traffic Volume",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- MIDDLE SECTION: MODEL COMPARISON ---
st.subheader("🏆 Model Leaderboard")
if metrics and 'rf' in metrics and 'lstm' in metrics:
    rf_rmse, rf_mae = metrics['rf']['rmse'], metrics['rf']['mae']
    lstm_rmse, lstm_mae = metrics['lstm']['rmse'], metrics['lstm']['mae']
    
    winner = "Random Forest" if rf_rmse < lstm_rmse else "LSTM"
    st.markdown(f"Based on the Root Mean Squared Error (RMSE) on the test dataset, the **<span class='winner-text'>{winner}</span>** model performs better!", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Random Forest RMSE", f"{rf_rmse:.2f}", delta="Winner" if winner == "Random Forest" else None, delta_color="normal")
    m2.metric("Random Forest MAE", f"{rf_mae:.2f}")
    m3.metric("LSTM RMSE", f"{lstm_rmse:.2f}", delta="Winner" if winner == "LSTM" else None, delta_color="normal")
    m4.metric("LSTM MAE", f"{lstm_mae:.2f}")
else:
    st.info("Train the models (`python train_models.py`) to see the performance leaderboard.")

st.divider()

# --- BOTTOM SECTION: PREDICTIONS ---
st.subheader("🔮 Run Live Predictions")

def display_prediction(pred_val):
    cat = get_traffic_category(pred_val, df)
    cat_class = f"badge-{cat.lower()}"
    st.markdown(f"### <span style='font-size:32px; font-weight:bold; color:#2c3e50;'>{int(pred_val)}</span> vehicles", unsafe_allow_html=True)
    st.markdown(f"Traffic Intensity: <span class='{cat_class}'>{cat}</span>", unsafe_allow_html=True)

pred_col1, pred_col2 = st.columns(2, gap="large")

with pred_col1:
    st.markdown("### 🌲 Random Forest Details")
    st.info("Uses time of day and recent hourly history.")
    
    scenarios = {
        "Custom": {"hour": 12, "day": 2, "l1": 500, "l2": 480, "l3": 450},
        "Monday Morning Rush Hour": {"hour": 8, "day": 0, "l1": 850, "l2": 820, "l3": 750},
        "Sunday Night": {"hour": 23, "day": 6, "l1": 150, "l2": 180, "l3": 210},
        "Midday Weekday": {"hour": 13, "day": 3, "l1": 400, "l2": 420, "l3": 450}
    }
    
    selected_scenario = st.selectbox("Scenario Presets", list(scenarios.keys()), key="rf_preset")
    config = scenarios[selected_scenario]
    
    with st.form("rf_form"):
        hour = st.slider("Hour of Day (0-23)", 0, 23, config["hour"])
        day = st.slider("Day of Week (0-6)", 0, 6, config["day"])
        
        c1, c2, c3 = st.columns(3)
        with c1: lag_1 = st.number_input("1 hr ago", value=config["l1"], min_value=0)
        with c2: lag_2 = st.number_input("2 hrs ago", value=config["l2"], min_value=0)
        with c3: lag_3 = st.number_input("3 hrs ago", value=config["l3"], min_value=0)
            
        rf_submit = st.form_submit_button("Predict Traffic Volume", type="primary")

    if rf_submit:
        if rf_model is not None:
            is_weekend = 1 if day >= 5 else 0
            features = np.array([[hour, day, is_weekend, lag_1, lag_2, lag_3]])
            prediction = rf_model.predict(features)[0]
            display_prediction(prediction)

with pred_col2:
    st.markdown("### 🧠 LSTM Deep Learning Model")
    st.info("Predict based on the last 24 continuous traffic volume readings.")
    
    default_seq = "0" * 24
    if len(df) >= 24:
        default_seq = ",".join(map(str, df['traffic_volume'].iloc[-24:].tolist()))
        
    if 'lstm_seq' not in st.session_state:
        st.session_state.lstm_seq = default_seq
        
    if st.button("Load Sample Sequence", help="Load 24 sequential values from the end of the dataset"):
        st.session_state.lstm_seq = default_seq

    with st.form("lstm_form"):
        sequence_input = st.text_area(
            "Enter 24 comma-separated values:",
            value=st.session_state.lstm_seq,
            height=105
        )
        lstm_submit = st.form_submit_button("Predict Next Value", type="primary")

    if lstm_submit:
        st.session_state.lstm_seq = sequence_input
        if lstm_model is not None and scaler is not None:
            try:
                values = [float(x.strip()) for x in sequence_input.split(',')]
                if len(values) != 24:
                    st.error(f"Expected exactly 24 values, but got {len(values)}.")
                else:
                    seq_array = np.array(values).reshape(-1, 1)
                    seq_scaled = scaler.transform(seq_array)
                    seq_reshaped = seq_scaled.reshape(1, 24, 1)
                    
                    pred_scaled = lstm_model.predict(seq_reshaped, verbose=0)
                    prediction = scaler.inverse_transform(pred_scaled)[0][0]
                    
                    display_prediction(prediction)
            except ValueError:
                st.error("Invalid input. Please ensure it's a comma-separated list of numbers.")
