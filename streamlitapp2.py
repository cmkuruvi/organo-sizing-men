import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------ PASSWORD -------------------
st.sidebar.header("🔒 Enter Password to Access Tool")
password = st.sidebar.text_input("Password", type="password")
VALID_PASSWORD = "ohdrog"
if password != VALID_PASSWORD:
    st.sidebar.warning("⚠️ Enter the correct password to proceed!")
    st.stop()

# ------------------ APP HEADER -----------------
st.image("2.png", width=200)
st.title("👖 AI-Powered Body Measurement Predictor - MEN")

# ------------------ DATA LOADING & MODEL -------------------
@st.cache_data
def load_data():
    file_path = 'Sizing Spreadsheet - Test Data - 13.05.2024.csv'
    df = pd.read_csv(file_path)
    return df

df = load_data()

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

st.markdown("#### Model Evaluation Metrics")
col1,col2,col3,col4 = st.columns(4)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R²", f"{r2:.2f}")

st.info("Open the sidebar (top left) to input your data.")

# ------------------ SIDEBAR INPUTS -------------------
st.sidebar.header("Enter Your Measurements")
weight = st.sidebar.number_input("Weight (kg)", min_value=40, max_value=200, value=78)
height = st.sidebar.number_input("Height (cm)", min_value=140, max_value=220, value=180)
chest = st.sidebar.number_input("Chest Value (0: Strong, 1: Average, 2: Wide)", min_value=0, max_value=2, value=1)
chest_hint = {0: "STRONG", 1: "AVERAGE", 2: "WIDE"}
st.sidebar.info(f"Hint: '{chest}' indicates {chest_hint[chest]}")

abdomen = st.sidebar.number_input("Abdomen Value (0: Flat, 1: Average, 2: Belly, 3: Belly+)", min_value=0, max_value=3, value=1)
abdomen_hint = {0: "FLAT", 1: "AVERAGE", 2: "BELLY", 3: "BELLY+"}
st.sidebar.info(f"Hint: '{abdomen}' indicates {abdomen_hint[abdomen]}")

st.sidebar.markdown("---")
st.sidebar.header("Fit & Build Preferences")

fit_type = st.sidebar.selectbox(
    "Preferred Fit", ["Regular", "Slim", "Relaxed/Athletic"], index=0,
    help="Regular: classic fit.\nSlim: trimmer thighs/bum.\nAthletic/Relaxed: more room at seat/thigh."
)
taper_type = st.sidebar.selectbox(
    "Taper",
    ["Straight (classic)", "Tapered (narrower at ankle)", "Bootcut/Loose"],
    index=0,
    help="Taper affects leg opening: Tapered is narrower than Straight; Bootcut/Loose is wider."
)
# For build type if desired (optional)
build_type = st.sidebar.selectbox(
    "Build Type", ["Average", "Stocky/Muscular", "Lean/Slim"],
    index=0,
    help="Further modify thigh and bum fit."
)

st.sidebar.markdown("---")
st.sidebar.header("Advanced")
custom_thigh = st.sidebar.number_input(
    "Your measured thigh (cm, optional)", min_value=0.0, max_value=100.0, step=0.5, value=0.0,
    help="Enter your true thigh circumference for feedback."
)
custom_leg_opening = st.sidebar.number_input(
    "Preferred leg opening (cm, optional)", min_value=0.0, max_value=80.0, step=0.5, value=0.0,
    help="Enter your ideal ankle/hem width."
)

# ------------------ FIT & TAPER ADJUSTMENTS -------------------
def adjust_for_fit_taper(waist, hips, thigh, rise, fit, taper, build):
    # Fit adjustments (cm)
    fit_adj = {
        "Regular": [0, 0, 0, 0],
        "Slim": [-2, -2, -1.5, -0.5],
        "Relaxed/Athletic": [+2, +2, +2, +0.5],
    }
    build_adj = {
        "Average": [0, 0, 0, 0],
        "Stocky/Muscular": [0, +1, +2.5, 0.7],
        "Lean/Slim": [0, -1, -1, 0],
    }
    # Leg opening as % of thigh (empirical menswear standards)
    taper_ratio = {
        "Straight (classic)": {"Regular": 0.72, "Slim": 0.68, "Relaxed/Athletic": 0.76},
        "Tapered (narrower at ankle)": {"Regular": 0.68, "Slim": 0.60, "Relaxed/Athletic": 0.72},
        "Bootcut/Loose": {"Regular": 0.84, "Slim": 0.80, "Relaxed/Athletic": 0.90}
    }
    # Apply sequentially
    w,h,t,r = waist, hips, thigh, rise
    for delta, bdelta in zip(fit_adj[fit], build_adj[build]):
        w += delta + bdelta
        h += delta + bdelta
        t += delta + bdelta
        r += delta + bdelta * 0.3  # rise less affected by muscularity

    adj_waist = w
    adj_hips = h
    adj_thigh = t
    adj_rise = r
    adj_leg_opening = adj_thigh * taper_ratio[taper][fit]
    return adj_waist, adj_hips, adj_thigh, adj_rise, adj_leg_opening

# ------------------ PREDICT & DISPLAY -------------------
if st.sidebar.button("Predict Measurements"):
    new_data = pd.DataFrame({'Weight': [weight],
                             'Height': [height],
                             'Chest': [chest],
                             'Abdomen': [abdomen]})
    predicted_values = model.predict(new_data)
    # Unpack
    pred = lambda idx: predicted_values[0,idx]
    pred_neck = pred(0)
    pred_sleeve = pred(1)
    pred_shoulder = pred(2)
    pred_chest = pred(3)
    pred_stomach = pred(4)
    pred_torso = pred(5)
    pred_bicep = pred(6)
    pred_wrist = pred(7)
    pred_rise = pred(8)
    pred_leg_length = pred(9)
    pred_waist = pred(10)
    pred_hips = pred(11)
    pred_thigh = pred(12)

    # ---- Apply adjustments ----
    adj_waist, adj_hips, adj_thigh, adj_rise, adj_leg_opening = adjust_for_fit_taper(
        pred_waist, pred_hips, pred_thigh, pred_rise, fit_type, taper_type, build_type
    )

    # ------ UI Tabs for Output -------
    tabs = st.tabs(["Full-Sleeve Shirts", "Pants", "Shorts", "Fit Analysis"])
    with tabs[0]:
        st.subheader("Predicted Full-Sleeve Shirt Measurements")
        st.metric("Neck", f"{pred_neck/2.54:.1f}\" | {pred_neck:.1f} cm")
        st.metric("Sleeve Length", f"{pred_sleeve/2.54:.1f}\" | {pred_sleeve:.1f} cm")
        st.metric("Shoulder Width", f"{pred_shoulder/2.54:.1f}\" | {pred_shoulder:.1f} cm")
        st.metric("Chest Around", f"{pred_chest/2.54:.1f}\" | {pred_chest:.1f} cm")
        st.metric("Stomach", f"{pred_stomach/2.54:.1f}\" | {pred_stomach:.1f} cm")
        st.metric("Torso Length", f"{pred_torso/2.54-1.5:.1f}\" | {pred_torso-3.8:.1f} cm")
        st.metric("Bicep", f"{pred_bicep/2.54:.1f}\" | {pred_bicep:.1f} cm")
        st.metric("Wrist", f"{pred_wrist/2.54:.1f}\" | {pred_wrist:.1f} cm")

    with tabs[1]:
        st.subheader(f"Predicted Pants ({fit_type}, {taper_type}, {build_type})")
        st.metric("Rise", f"{adj_rise/2.54:.1f}\" | {adj_rise:.1f} cm", help="Distance from crotch to waist band")
        st.metric("Leg Length", f"{(pred_leg_length-3.8)/2.54:.1f}\" | {pred_leg_length-3.8:.1f} cm")
        st.metric("Waist", f"{adj_waist/2.54:.1f}\" | {adj_waist:.1f} cm")
        st.metric("Hips", f"{adj_hips/2.54:.1f}\" | {adj_hips:.1f} cm")
        st.metric("Thigh", f"{adj_thigh/2.54:.1f}\" | {adj_thigh:.1f} cm")
        st.metric("Leg Opening", f"{adj_leg_opening/2.54:.1f}\" | {adj_leg_opening:.1f} cm", help="Width at the pant hem")
        # Optionally check custom input
        if custom_thigh > 0:
            thigh_diff = abs(adj_thigh - custom_thigh)
            if thigh_diff <= 2:
                st.success(f"Predicted thigh ({adj_thigh:.1f}cm) is within 2cm of your measurement ({custom_thigh:.1f}cm).")
            else:
                st.warning(f"Predicted thigh differs by {thigh_diff:.1f}cm from your supplied ({custom_thigh:.1f}cm).")
        if custom_leg_opening > 0:
            leg_diff = abs(adj_leg_opening - custom_leg_opening)
            if leg_diff <= 1.5:
                st.success(f"Predicted leg opening ({adj_leg_opening:.1f}cm) very close to your input ({custom_leg_opening:.1f}cm).")
            else:
                st.warning(f"Leg opening predicted ({adj_leg_opening:.1f}cm) differs by {leg_diff:.1f}cm from your preference ({custom_leg_opening:.1f}cm).")

    with tabs[2]:
        st.subheader("Predicted Shorts Measurements")
        shorts_length = (pred_leg_length - 25)
        shorts_leg_opening = adj_thigh * 0.75
        st.metric("Shorts Leg Length", f"{shorts_length/2.54:.1f}\" | {shorts_length:.1f} cm")
        st.metric("Shorts Leg Opening", f"{shorts_leg_opening/2.54:.1f}\" | {shorts_leg_opening:.1f} cm", help="Width at leg hem for shorts")

    with tabs[3]:
        st.subheader("Fit Analysis & Custom Guidance")
        st.write(f"**Fit:** {fit_type}, **Taper:** {taper_type}, **Build:** {build_type}")
        st.write("- Try 'Relaxed/Athletic' for sports/muscular builds.")
        st.write("- For ample seat and thigh, choose 'Relaxed/Athletic' and 'Straight' or 'Bootcut/Loose' tapers.")
        st.write("- For trimmer look, choose 'Slim' and 'Tapered'.")
        st.info("You can measure your thigh at its widest point; leg opening is the flat measurement at the pant hem, doubled for circumference.")

    st.markdown(
        "*Predictions include fit, build, and taper adjustments, giving more accurate recommendations. Feel free to adjust and compare to your own measured numbers.*"
    )
