import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Password Authentication
st.sidebar.header("🔒 Enter Password to Access Tool")

password = st.sidebar.text_input("Password", type="password")

# Define your valid password (Change this to your desired password)
VALID_PASSWORD = "ohdrog"

# Check password before allowing access
if password != VALID_PASSWORD:
    st.sidebar.warning("⚠️ Enter the correct password to proceed!")
    st.stop()  # Stops the app from loading further

st.image("2.png", width=200)
st.title("👔 AI-Powered Body Measurement Predictor - MEN")

# Function to load the dataset
@st.cache_data
def load_data():
    file_path = 'Sizing Spreadsheet - Test Data - 13.05.2024.csv'
    df = pd.read_csv(file_path)
    return df

# Load the dataset and display a preview
df = load_data()

# Function to train the linear regression model
@st.cache_data
def train_model(data):
    # Features: Weight, Height, Chest, and Abdomen
    X = data[['Weight', 'Height', 'Chest', 'Abdomen']]
    # Targets: multiple measurements
    y = data[['Neck', 'Sleeve Length (F/S)', 'Shoulder Width', 'Chest Around',
              'Stomach', 'Torso Length', 'Bicep', 'Wrist', 'Rise',
              'Length (Leg)', 'Waist (Pants)', 'Hips', 'THigh']]
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set and evaluate performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5  # calculate RMSE manually
    r2 = r2_score(y_test, y_pred)
    return model, mae, mse, rmse, r2

# Train the model and display performance metrics
model, mae, mse, rmse, r2 = train_model(df)
st.subheader("Model Evaluation Metrics")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Root Mean Squared Error: {rmse:.2f}")
st.write(f"R-squared: {r2:.2f}")

st.subheader("OPEN SIDEBAR -button at top left- to input values.")

# Sidebar inputs for new measurements
st.sidebar.header("Enter Your Measurements")
weight = st.sidebar.number_input("Weight (kg)", min_value=40, value=68)
height = st.sidebar.number_input("Height (cm)", min_value=140, value=178)

# Updated chest input with hint
chest = st.sidebar.number_input("Chest Value (0: Strong, 1: Average, 2: Wide)", min_value=0, max_value=2, value=1)
if chest == 0:
    st.sidebar.info("Hint: '0' indicates STRONG")
elif chest == 1:
    st.sidebar.info("Hint: '1' indicates AVERAGE")
elif chest == 2:
    st.sidebar.info("Hint: '2' indicates WIDE")

# Updated abdomen input with hint
abdomen = st.sidebar.number_input("Abdomen Value (0: Flat, 1: Average, 2: Belly, 3: Belly+)", min_value=0, max_value=3, value=1)
if abdomen == 0:
    st.sidebar.info("Hint: '0' indicates FLAT")
elif abdomen == 1:
    st.sidebar.info("Hint: '1' indicates AVERAGE")
elif abdomen == 2:
    st.sidebar.info("Hint: '2' indicates BELLY")
elif abdomen == 3:
    st.sidebar.info("Hint: '3' indicates BELLY+")

if st.sidebar.button("Predict Measurements"):
    # Create a DataFrame for the new data point
    new_data = pd.DataFrame({
        'Weight': [weight],
        'Height': [height],
        'Chest': [chest],
        'Abdomen': [abdomen]
    })
    # Get predictions
    predicted_values = model.predict(new_data)
    
    # Extract individual predicted measurements
    pred_neck = predicted_values[:, 0]
    pred_sleeve = predicted_values[:, 1]
    pred_shoulder = predicted_values[:, 2]
    pred_chest = predicted_values[:, 3]
    pred_stomach = predicted_values[:, 4]
    pred_torso = predicted_values[:, 5]
    pred_bicep = predicted_values[:, 6]
    pred_wrist = predicted_values[:, 7]
    pred_rise = predicted_values[:, 8]
    pred_leg_length = predicted_values[:, 9]
    pred_waist = predicted_values[:, 10]
    pred_hips = predicted_values[:, 11]
    pred_thigh = predicted_values[:, 12]
    
    # Display predictions for full-sleeve shirts
    st.subheader("Predicted Measurements for Full Sleeve Shirts")
    st.write(f"Neck Measure: IN: {round(pred_neck[0] / 2.54, 1)} , CM: {round(pred_neck[0], 1)}")
    st.write(f"Sleeve Length: IN: {round(pred_sleeve[0] / 2.54, 1)} , CM: {round(pred_sleeve[0], 1)}")
    st.write(f"Shoulder Width: IN: {round(pred_shoulder[0] / 2.54, 1)} , CM: {round(pred_shoulder[0], 1)}")
    st.write(f"Chest Around: IN: {round(pred_chest[0] / 2.54, 1)} , CM: {round(pred_chest[0], 1)}")
    st.write(f"Stomach: IN: {round(pred_stomach[0] / 2.54, 1)} , CM: {round(pred_stomach[0], 1)}")
    st.write(f"Torso Length: IN: {round(pred_torso[0] / 2.54 - 1.5, 1)} , CM: {round(pred_torso[0] - 3.81, 1)}")
    st.write(f"Bicep: IN: {round(pred_bicep[0] / 2.54, 1)} , CM: {round(pred_bicep[0], 1)}")
    st.write(f"Wrist: IN: {round(pred_wrist[0] / 2.54, 1)} , CM: {round(pred_wrist[0], 1)}")
    st.write(f"Hips: IN: {round(pred_hips[0] / 2.54, 1)} , CM: {round(pred_hips[0], 1)}")
    
    # Display predictions for pants measurements
    st.subheader("Predicted Measurements for Pants")
    st.write(f"Rise: IN: {round(pred_rise[0] / 2.54, 1)} , CM: {round(pred_rise[0], 1)}")
    st.write(f"Leg Length: IN: {round(pred_leg_length[0] / 2.54 - 1.5, 1)} , CM: {round(pred_leg_length[0] - 3.8, 1)}")
    st.write(f"Waist: IN: {round(pred_waist[0] / 2.54, 1)} , CM: {round(pred_waist[0], 1)}")
    st.write(f"Hips: IN: {round(pred_hips[0] / 2.54, 1)} , CM: {round(pred_hips[0], 1)}")
    st.write(f"Thighs: IN: {round(pred_thigh[0] / 2.54, 1)} , CM: {round(pred_thigh[0], 1)}")
    
    # Display predictions for shorts measurements
    st.subheader("Predicted Measurements for Shorts")
    st.write(f"Shorts Leg Length: IN: {round(pred_leg_length[0] / 2.54 - 22, 1)} , CM: {round(pred_leg_length[0] - 58, 1)}")
    st.write(f"Half Sleeve: IN: {round(pred_sleeve[0] / (2.54 * 2.5), 1)} , CM: {round(pred_sleeve[0], 1)}")
