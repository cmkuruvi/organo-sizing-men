import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

st.title("👔 AI-Powered Body Measurement Predictor - MEN")

# Load the dataset (caching to avoid reloading on every interaction)
@st.cache_data
def load_data():
    file_path = 'Sizing Spreadsheet - Test Data - 13.05.2024.csv'
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Train the model (cache to avoid re-training on every interaction)
@st.cache_data
def train_model(data):
    X = data[['Weight', 'Height', 'Chest', 'Abdomen']]
    y = data[['Neck', 'Sleeve Length (F/S)', 'Shoulder Width', 'Chest Around', 'Stomach',
              'Torso Length', 'Bicep', 'Wrist', 'Rise', 'Length (Leg)',
              'Waist (Pants)', 'Hips', 'THigh']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Model evaluation (optional display)
    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

model, mae, mse, rmse, r2 = train_model(df)

# Sidebar inputs for new data
st.sidebar.header("Input Your Measurements")
weight = st.sidebar.number_input("Weight (kg)", min_value=50, max_value=130, step=1)
height = st.sidebar.number_input("Height (cm)", min_value=150, max_value=220, step=1)
chest = st.sidebar.number_input("Chest Value", min_value=0, max_value=3, step=1)
abdomen = st.sidebar.number_input("Abdomen Value", min_value=0, max_value=3, step=1)

if st.sidebar.button("Predict Measurements"):
    new_data = pd.DataFrame({'Weight': [weight],
                             'Height': [height],
                             'Chest': [chest],
                             'Abdomen': [abdomen]})
    predicted_values = model.predict(new_data)
    
    # Extract predicted measurements
    predicted_neck = predicted_values[:, 0]
    predicted_sleeve = predicted_values[:, 1]
    predicted_shoulder = predicted_values[:, 2]
    predicted_chest = predicted_values[:, 3]
    predicted_stomach = predicted_values[:, 4]
    predicted_torso = predicted_values[:, 5]
    predicted_bicep = predicted_values[:, 6]
    predicted_wrist = predicted_values[:, 7]
    predicted_rise = predicted_values[:, 8]
    predicted_leg = predicted_values[:, 9]
    predicted_waist = predicted_values[:, 10]
    predicted_hips = predicted_values[:, 11]
    predicted_thigh = predicted_values[:, 12]

    st.subheader("Measurements for Full Sleeve Shirts")
    st.write(f"Predicted Neck: IN: {round(predicted_neck[0]/2.54, 1)} , CM: {round(predicted_neck[0], 1)}")
    st.write(f"Predicted Sleeve: IN: {round(predicted_sleeve[0]/2.54, 1)} , CM: {round(predicted_sleeve[0], 1)}")
    st.write(f"Predicted Shoulder: IN: {round(predicted_shoulder[0]/2.54, 1)} , CM: {round(predicted_shoulder[0], 1)}")
    st.write(f"Predicted Chest: IN: {round(predicted_chest[0]/2.54, 1)} , CM: {round(predicted_chest[0], 1)}")
    st.write(f"Predicted Stomach: IN: {round(predicted_stomach[0]/2.54, 1)} , CM: {round(predicted_stomach[0], 1)}")
    st.write(f"Predicted Torso Length: IN: {round(predicted_torso[0]/2.54 - 1.5, 1)} , CM: {round(predicted_torso[0]-3.81, 1)}")
    st.write(f"Predicted Bicep: IN: {round(predicted_bicep[0]/2.54, 1)} , CM: {round(predicted_bicep[0], 1)}")
    st.write(f"Predicted Wrist: IN: {round(predicted_wrist[0]/2.54, 1)} , CM: {round(predicted_wrist[0], 1)}")
    st.write(f"Predicted Hips: IN: {round(predicted_hips[0]/2.54, 1)} , CM: {round(predicted_hips[0], 1)}")

    st.subheader("Measurements for Pants")
    st.write(f"Predicted Rise: IN: {round(predicted_rise[0]/2.54, 1)} , CM: {round(predicted_rise[0], 1)}")
    st.write(f"Predicted Leg Length: IN: {round(predicted_leg[0]/2.54 - 1.5, 1)} , CM: {round(predicted_leg[0]-3.8, 1)}")
    st.write(f"Predicted Waist: IN: {round(predicted_waist[0]/2.54, 1)} , CM: {round(predicted_waist[0], 1)}")
    st.write(f"Predicted Hips: IN: {round(predicted_hips[0]/2.54, 1)} , CM: {round(predicted_hips[0], 1)}")
    st.write(f"Predicted Thigh: IN: {round(predicted_thigh[0]/2.54, 1)} , CM: {round(predicted_thigh[0], 1)}")

    st.subheader("Measurements for Shorts")
    st.write(f"Predicted Shorts Leg Length: IN: {round(predicted_leg[0]/2.54 - 22, 1)} , CM: {round(predicted_leg[0]-58, 1)}")
    st.write(f"Predicted Half Sleeve: IN: {round(predicted_sleeve[0]/2.54/2.5, 1)} , CM: {round(predicted_sleeve[0], 1)}")
