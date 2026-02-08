"""
Flat Price Prediction App
CIS6005 Computational Intelligence

A Streamlit application for predicting flat prices based on property features.
Run with: streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Flat Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load models and preprocessors
@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects."""
    try:
        models_dir = "models"
        data_dir = os.path.join("data", "processed")

        xgb_model = joblib.load(os.path.join(models_dir, "xgboost_model.pkl"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        label_encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
        feature_cols = joblib.load(os.path.join(data_dir, "feature_cols.pkl"))

        return xgb_model, scaler, label_encoders, feature_cols
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the notebooks first to generate model files.")
        return None, None, None, None

# Load models
xgb_model, scaler, label_encoders, feature_cols = load_models()

# Title and description
st.title("🏠 Flat Price Prediction System")
st.markdown("""
This application predicts flat prices in Saint Petersburg using a trained XGBoost model.
Enter the property features below to get an estimated price.
""")

# Only show the app if models are loaded
if xgb_model is not None:
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown(f"""
        **Model Type:** XGBoost Regressor
        **Features:** {len(feature_cols)}
        **Performance (Validation):**
        - RMSE: 230,083 rubles
        - MAE: 178,699 rubles
        - R²: 0.9984
        """)

        st.markdown("---")
        st.markdown("**Top Features:**")
        st.markdown("""
        1. Total Area
        2. Other Area
        3. Rooms Count
        4. Bath Area
        5. Kitchen Area
        """)

    # Main content area - Input form
    st.header("Property Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Area Information (sq m)")
        kitchen_area = st.number_input("Kitchen Area", min_value=1, max_value=30, value=10, step=1)
        bath_area = st.number_input("Bath Area", min_value=1, max_value=36, value=15, step=1)
        other_area = st.number_input("Other Area", min_value=11.0, max_value=91.0, value=20.0, step=0.5)
        total_area = st.number_input("Total Area", min_value=25.0, max_value=200.0, value=50.0, step=0.5)
        extra_area = st.number_input("Extra Area (Balcony/Loggia)", min_value=0, max_value=20, value=5, step=1)

    with col2:
        st.subheader("Property Details")
        rooms_count = st.number_input("Number of Rooms", min_value=0, max_value=9, value=2, step=1)
        bath_count = st.selectbox("Number of Bathrooms", [1, 2, 3], index=0)
        year = st.number_input("Year of Construction", min_value=1900, max_value=2020, value=2000, step=1)
        ceil_height = st.number_input("Ceiling Height (m)", min_value=2.5, max_value=5.0, value=2.7, step=0.1)
        floor = st.number_input("Floor Number", min_value=1, max_value=23, value=5, step=1)
        floor_max = st.number_input("Total Floors in Building", min_value=1, max_value=23, value=10, step=1)

    with col3:
        st.subheader("Amenities & Location")
        gas = st.selectbox("Gas", ["No", "Yes"], index=1)
        hot_water = st.selectbox("Hot Water", ["No", "Yes"], index=1)
        central_heating = st.selectbox("Central Heating", ["No", "Yes"], index=1)
        extra_area_count = st.selectbox("Extra Area Count", [0, 1, 2], index=1)
        extra_area_type = st.selectbox(
            "Extra Area Type",
            list(label_encoders['extra_area_type_name'].classes_),
            index=0
        )
        district = st.selectbox(
            "District",
            list(label_encoders['district_name'].classes_),
            index=0
        )

    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_data = {
                'kitchen_area': kitchen_area,
                'bath_area': bath_area,
                'other_area': other_area,
                'gas': 1 if gas == "Yes" else 0,
                'hot_water': 1 if hot_water == "Yes" else 0,
                'central_heating': 1 if central_heating == "Yes" else 0,
                'extra_area': extra_area,
                'extra_area_count': extra_area_count,
                'year': year,
                'ceil_height': ceil_height,
                'floor_max': floor_max,
                'floor': floor,
                'total_area': total_area,
                'bath_count': bath_count,
                'extra_area_type_name': int(label_encoders['extra_area_type_name'].transform([extra_area_type])[0]),
                'district_name': int(label_encoders['district_name'].transform([district])[0]),
                'rooms_count': rooms_count
            }

            # Create feature array in correct order
            feature_array = np.array([[input_data[col] for col in feature_cols]])

            # Make prediction
            prediction = xgb_model.predict(feature_array)[0]

            # Display results
            st.success("Prediction Complete!")

            # Show prediction in a nice format
            st.markdown("### Estimated Price")
            st.markdown(f"# ₽ {prediction:,.0f}")
            st.caption(f"Approximately ${prediction/75:,.0f} USD (at ₽75/$1)")

            # Show input summary
            with st.expander("📋 View Input Summary"):
                summary_df = pd.DataFrame({
                    'Feature': ['Total Area', 'Rooms', 'District', 'Year Built', 'Floor', 'Ceiling Height'],
                    'Value': [
                        f"{total_area} sq m",
                        rooms_count,
                        district,
                        year,
                        f"{floor} / {floor_max}",
                        f"{ceil_height} m"
                    ]
                })
                st.table(summary_df)

        except Exception as e:
            st.error(f"Error making prediction: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    CIS6005 Computational Intelligence - Flat Price Prediction<br>
    Model: XGBoost Regressor | Dataset: ITMO Saint Petersburg Housing Data
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Models not loaded. Please run the Jupyter notebooks to train and save the models first.")
    st.markdown("""
    ### Steps to get started:
    1. Run `01_EDA.ipynb` for data exploration
    2. Run `02_Preprocessing.ipynb` to prepare the data
    3. Run `03_Model_Training.ipynb` to train models
    4. Restart this app
    """)
