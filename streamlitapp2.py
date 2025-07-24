import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Password Authentication ---
st.sidebar.header("🔒 Enter Password to Access Tool")
password = st.sidebar.text_input("Password", type="password")
VALID_PASSWORD = "ohdrog"
if password != VALID_PASSWORD:
    st.sidebar.warning("⚠️ Enter the correct password to proceed!")
    st.stop()

st.image("2.png", width=200)
st.title("👔 AI-Powered Body Measurement Predictor - MEN")

# --- Data Loading ---
@st.cache_data
def load_data():
    file_path = 'Sizing Spreadsheet - Test Data - 13.05.2024.csv'
    df = pd.read_csv(file_path)
    return df

df = load_data()

# --- Model Training ---
@st.cache_data
def train_model(data):
    X = data[['Weight', 'Height', 'Chest', 'Abdomen']]
    y = data[['Neck', 'Sleeve Length (F/S)', 'Shoulder Width', 'Chest Around',
              'Stomach', 'Torso Length', 'Bicep', 'Wrist', 'Rise',
              'Length (Leg)', 'Waist (Pants)', 'Hips', 'THigh']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    return model, mae, mse, rmse, r2

model, mae, mse, rmse, r2 = train_model(df)
st.subheader("Model Evaluation Metrics")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Root Mean Squared Error: {rmse:.2f}")
st.write(f"R-squared: {r2:.2f}")

st.subheader("OPEN SIDEBAR -button at top left- to input values.")

# --- Sidebar Inputs ---
st.sidebar.header("Enter Your Measurements")
weight = st.sidebar.number_input("Weight (kg)", min_value=40, value=68)
height = st.sidebar.number_input("Height (cm)", min_value=140, value=178)
chest = st.sidebar.number_input("Chest Value (0: Strong, 1: Average, 2: Wide)", min_value=0, max_value=2, value=1)
if chest == 0:
    st.sidebar.info("Hint: '0' indicates STRONG")
elif chest == 1:
    st.sidebar.info("Hint: '1' indicates AVERAGE")
elif chest == 2:
    st.sidebar.info("Hint: '2' indicates WIDE")
abdomen = st.sidebar.number_input("Abdomen Value (0: Flat, 1: Average, 2: Belly, 3: Belly+)", min_value=0, max_value=3, value=1)
if abdomen == 0:
    st.sidebar.info("Hint: '0' indicates FLAT")
elif abdomen == 1:
    st.sidebar.info("Hint: '1' indicates AVERAGE")
elif abdomen == 2:
    st.sidebar.info("Hint: '2' indicates BELLY")
elif abdomen == 3:
    st.sidebar.info("Hint: '3' indicates BELLY+")

# --- New Fit and Taper Selectors ---
fit_type = st.sidebar.selectbox("Fit Type", ["Regular", "Slim"], index=0)
taper_type = st.sidebar.selectbox("Taper", ["Straight", "Tapered"], index=0)

def adjust_for_fit_taper(waist, hips, thigh, fit, taper):
    # Fit adjustments (cm): Regular (none), Slim (trimmer)
    fit_adj = {"Regular": [0, 0, 0], "Slim": [-2, -2, -1]}
    # Taper ratios (Leg Opening = ratio * thigh)
    taper_ratio = {"Straight": {"Regular": 0.75, "Slim": 0.68},
                   "Tapered": {"Regular": 0.65, "Slim": 0.58}}
    waist_adj, hips_adj, thigh_adj = fit_adj[fit]
    adj_waist = waist + waist_adj
    adj_hips = hips + hips_adj
    adj_thigh = thigh + thigh_adj
    adj_leg_opening = adj_thigh * taper_ratio[taper][fit]
    return adj_waist, adj_hips, adj_thigh, adj_leg_opening

if st.sidebar.button("Predict Measurements"):
    new_data = pd.DataFrame({'Weight': [weight],
                             'Height': [height],
                             'Chest': [chest],
                             'Abdomen': [abdomen]})
    predicted_values = model.predict(new_data)
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

    # --- Apply Fit & Taper Adjustments ---
    adj_waist, adj_hips, adj_thigh, adj_leg_opening = adjust_for_fit_taper(
        pred_waist[0], pred_hips[0], pred_thigh[0], fit_type, taper_type)

    # --- Display predictions for full-sleeve shirts ---
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

    # --- Display predictions for pants, with adjustments and Leg Opening ---
    st.subheader(f"Predicted Measurements for Pants ({fit_type} Fit, {taper_type})")
    st.write(f"Rise: IN: {round(pred_rise[0] / 2.54, 1)} , CM: {round(pred_rise[0], 1)}")
    st.write(f"Leg Length: IN: {round(pred_leg_length[0] / 2.54 - 1.5, 1)} , CM: {round(pred_leg_length[0] - 3.8, 1)}")
    st.write(f"Waist: IN: {round(adj_waist / 2.54, 1)} , CM: {round(adj_waist, 1)}")
    st.write(f"Hips: IN: {round(adj_hips / 2.54, 1)} , CM: {round(adj_hips, 1)}")
    st.write(f"Thighs: IN: {round(adj_thigh / 2.54, 1)} , CM: {round(adj_thigh, 1)}")
    st.write(f"Leg Opening: IN: {round(adj_leg_opening / 2.54, 1)} , CM: {round(adj_leg_opening, 1)}")

    # --- Display predictions for shorts measurements ---
    st.subheader("Predicted Measurements for Shorts")
    st.write(f"Shorts Leg Length: IN: {round(pred_leg_length[0] / 2.54 - 22, 1)} , CM: {round(pred_leg_length[0] - 58, 1)}")
    st.write(f"Half Sleeve: IN: {round(pred_sleeve[0] / (2.54 * 2.5), 1)} , CM: {round(pred_sleeve[0], 1)}")
