import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from train_model import load_and_preprocess_data, train_and_evaluate, save_model

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #00d2ff;
        color: #000;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
        color: #fff;
        transform: scale(1.02);
    }
    .metric-card {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d2ff;
    }
    .metric-label {
        font-size: 1rem;
        color: #b0bec5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .header-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "xgboost_air_quality_model.pkl"

# --- Helper Functions ---
@st.cache_resource
def load_saved_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

# --- Main App Structure ---
def main():
    st.markdown('<div class="header-container"><h1>🌍 Smart Air Quality Monitoring & Prediction</h1></div>', unsafe_allow_html=True)
    st.markdown("Predict pollution levels using machine learning based on real-time environmental data.")

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2911/2911956.png", width=100)
        st.title("Settings & Training")
        st.markdown("---")
        
        st.subheader("Model Management")
        if st.button("🚀 Retrain Model (Fetch latest data)"):
            with st.spinner("Fetching data and training XGBoost model..."):
                try:
                    X, y, features, target_col = load_and_preprocess_data()
                    model, metrics = train_and_evaluate(X, y)
                    save_model(model, features, target_col, metrics, MODEL_PATH)
                    st.success("Model trained and saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        
        st.markdown("---")
        st.info("This system uses the UCI Machine Learning Repository Air Quality dataset. The default target is CO(GT).")

    # Load Model
    model_data = load_saved_model()

    if model_data is None:
        st.warning("⚠️ No trained model found. Please click 'Retrain Model' in the sidebar to fetch data and train the initial model.")
        return

    # Extract model info
    model = model_data['model']
    features = model_data['features']
    target_col = model_data['target_col']
    metrics = model_data['metrics']

    # --- Dashboard Layout ---
    tab1, tab2, tab3 = st.tabs(["🔮 Real-time Prediction", "📊 Model Insights", "📈 Feature Explorer"])

    with tab1:
        st.subheader(f"Predict {target_col} Level")
        st.markdown("Adjust the environmental sensor readings below to predict the pollution concentration.")
        
        # Create input fields dynamically based on features
        # We will group them in columns for a better layout
        cols = st.columns(3)
        input_data = {}
        
        # Reasonable default values for UCI Air Quality dataset features
        defaults = {
            'PT08.S1(CO)': 1000.0,
            'NMHC(GT)': 200.0,
            'C6H6(GT)': 10.0,
            'PT08.S2(NMHC)': 1000.0,
            'NOx(GT)': 250.0,
            'PT08.S3(NOx)': 800.0,
            'NO2(GT)': 100.0,
            'PT08.S4(NO2)': 1500.0,
            'PT08.S5(O3)': 1000.0,
            'T': 20.0,  # Temperature
            'RH': 50.0, # Relative Humidity
            'AH': 1.0   # Absolute Humidity
        }

        for i, feature in enumerate(features):
            col_idx = i % 3
            default_val = defaults.get(feature, 50.0)
            with cols[col_idx]:
                # Using number_input for precision
                input_data[feature] = st.number_input(f"{feature}", value=float(default_val), format="%.2f")

        st.markdown("---")
        
        # Predict Button
        if st.button("Predict Pollution Level", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            # Display result
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: rgba(0, 210, 255, 0.1); border: 2px solid #00d2ff; border-radius: 15px; margin-top: 20px;">
                <h2 style="color: #fafafa; margin-bottom: 10px;">Predicted {target_col} Concentration</h2>
                <h1 style="color: #00d2ff; font-size: 4rem; margin: 0;">{prediction:.2f}</h1>
                <p style="color: #b0bec5; margin-top: 10px;">Based on current environmental inputs.</p>
            </div>
            """, unsafe_allow_html=True)

            # Air Quality Index (AQI) - rough estimation for visual effect
            aqi_status = "Good"
            aqi_color = "#00e676" # Green
            if prediction > 2.0:
                aqi_status = "Moderate"
                aqi_color = "#ffea00" # Yellow
            if prediction > 4.0:
                aqi_status = "Unhealthy"
                aqi_color = "#ff3d00" # Red
                
            st.markdown(f"""
            <div style="text-align: center; margin-top: 15px;">
                <span style="font-size: 1.2rem; color: #fafafa;">Estimated Air Quality: </span>
                <span style="font-size: 1.5rem; font-weight: bold; color: {aqi_color};">{aqi_status}</span>
            </div>
            """, unsafe_allow_html=True)


    with tab2:
        st.subheader("Model Performance Metrics")
        st.markdown("These metrics evaluate the accuracy of the XGBoost regression model.")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{metrics['R2']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">{metrics['RMSE']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MAE</div>
                <div class="metric-value">{metrics['MAE']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Feature Importance")
        st.markdown("Which sensors contribute most to predicting the pollution level?")
        
        # Get feature importances from XGBoost
        importance = model.feature_importances_
        imp_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=True)

        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='teal',
                     template='plotly_dark')
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Dataset Information")
        st.markdown("""
        **Data Source:** UCI Machine Learning Repository - Air Quality Dataset.
        
        The dataset contains responses from a gas multisensor device deployed on the field in an Italian city. Hourly responses averages are recorded along with gas concentrations references from a certified analyzer.
        
        **Features:**
        - **PT08.S1 (CO)**: Tin oxide hourly averaged sensor response (nominally CO targeted)
        - **NMHC (GT)**: True hourly averaged overall Non Metanic HydroCarbons concentration
        - **C6H6 (GT)**: True hourly averaged Benzene concentration
        - **PT08.S2 (NMHC)**: Titania hourly averaged sensor response (nominally NMHC targeted)
        - **NOx (GT)**: True hourly averaged NOx concentration
        - **PT08.S3 (NOx)**: Tungsten oxide hourly averaged sensor response (nominally NOx targeted)
        - **NO2 (GT)**: True hourly averaged NO2 concentration
        - **PT08.S4 (NO2)**: Tungsten oxide hourly averaged sensor response (nominally NO2 targeted)
        - **PT08.S5 (O3)**: Indium oxide hourly averaged sensor response (nominally O3 targeted)
        - **T**: Temperature in Â°C
        - **RH**: Relative Humidity (%)
        - **AH**: Absolute Humidity
        """)

if __name__ == "__main__":
    main()
