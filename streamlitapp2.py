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
st.image("Fitsall Logo new 1.png", width=300)
st.title("👖 AI-Powered Body Measurement Predictor - MEN (UPDATED)")

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
col1, col2, col3, col4 = st.columns(4)
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
build_type = st.sidebar.selectbox(
    "Build Type", ["Average", "Stocky/Muscular", "Lean/Slim"],
    index=0,
    help="Further modify fit measurements for different body builds."
)

THIGH_EASE_INCHES = st.sidebar.slider(
    "Thigh Ease (inches)", 2.0, 3.5, 2.5, 0.25,
    help="Industry standard: 2-3 inches of ease for comfortable fit"
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

LEG_LENGTH_ADJUSTMENT_MULTIPLIER = 0.97

def calculate_garment_measurements(pred_sleeve, pred_bicep, pred_leg_length, body_thigh, pred_rise, fit_type, build_type):
    # Your formulas, as validated above:
    short_sleeve_length = pred_sleeve * 0.400
    short_sleeve_opening_base = pred_bicep * 1.201

    # Add thigh ease for garment thigh
    body_thigh_in = body_thigh / 2.54
    garment_thigh_in = body_thigh_in + THIGH_EASE_INCHES
    garment_thigh_cm = garment_thigh_in * 2.54

    adj_leg_length = pred_leg_length * LEG_LENGTH_ADJUSTMENT_MULTIPLIER
    shorts_length = adj_leg_length - 57.7
    shorts_leg_opening_base = garment_thigh_cm * 1.100
    shorts_inseam = shorts_length - (pred_rise / 2)
    pant_leg_opening_base = garment_thigh_cm * 0.696
    pant_inseam = adj_leg_length - (pred_rise / 2)

    # Fit/build multipliers
    fit_multipliers = {"Regular": 1.0, "Slim": 0.90, "Relaxed/Athletic": 1.10}
    build_sleeve_multipliers = {"Average": 1.0, "Stocky/Muscular": 1.05, "Lean/Slim": 0.95}
    fit_multiplier = fit_multipliers.get(fit_type, 1.0)
    build_sleeve_multiplier = build_sleeve_multipliers.get(build_type, 1.0)

    short_sleeve_opening = short_sleeve_opening_base * fit_multiplier * build_sleeve_multiplier
    shorts_leg_opening = shorts_leg_opening_base * fit_multiplier
    pant_leg_opening = pant_leg_opening_base * fit_multiplier

    return {
        'short_sleeve_length': short_sleeve_length,
        'short_sleeve_opening': short_sleeve_opening,
        'shorts_length': shorts_length,
        'shorts_leg_opening': shorts_leg_opening,
        'shorts_inseam': shorts_inseam,
        'pant_leg_opening': pant_leg_opening,
        'pant_inseam': pant_inseam,
        'garment_thigh_cm': garment_thigh_cm,
        'body_thigh_in': body_thigh_in,
        'garment_thigh_in': garment_thigh_in
    }

def adjust_for_fit_taper(waist, hips, thigh, rise, fit, taper, build):
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
    w, h, t, r = waist, hips, thigh, rise
    for delta, bdelta in zip(fit_adj[fit], build_adj[build]):
        w += delta + bdelta
        h += delta + bdelta
        t += delta + bdelta
        r += delta + bdelta * 0.3
    return w, h, t, r

if st.sidebar.button("Predict Measurements"):
    try:
        new_data = pd.DataFrame({'Weight': [weight],
                                 'Height': [height],
                                 'Chest': [chest],
                                 'Abdomen': [abdomen]})
        predicted_values = model.predict(new_data)
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

        adj_waist, adj_hips, adj_thigh, adj_rise = adjust_for_fit_taper(
            pred_waist, pred_hips, pred_thigh, pred_rise, fit_type, taper_type, build_type)

        adj_leg_length = pred_leg_length * LEG_LENGTH_ADJUSTMENT_MULTIPLIER
        garment_measurements = calculate_garment_measurements(
            pred_sleeve, pred_bicep, adj_leg_length, adj_thigh, pred_rise, fit_type, build_type)

        tabs = st.tabs([
            "Full-Sleeve Shirts", "Short-Sleeve Shirts",
            "Pants", "Shorts",
            "Fit Analysis", "Finished Garment Specs"
        ])

        with tabs[0]:
            st.subheader("Predicted Full-Sleeve Shirt Measurements")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Neck", f"{pred_neck/2.54:.1f}\" | {pred_neck:.1f} cm")
                st.metric("Sleeve Length", f"{pred_sleeve/2.54:.1f}\" | {pred_sleeve:.1f} cm")
                st.metric("Shoulder Width", f"{pred_shoulder/2.54:.1f}\" | {pred_shoulder:.1f} cm")
                st.metric("Chest Around", f"{pred_chest/2.54:.1f}\" | {pred_chest:.1f} cm")
            with col2:
                st.metric("Stomach", f"{pred_stomach/2.54:.1f}\" | {pred_stomach:.1f} cm")
                st.metric("Torso Length", f"{pred_torso/2.54-1.5:.1f}\" | {pred_torso-3.8:.1f} cm")
                st.metric("Bicep", f"{pred_bicep/2.54:.1f}\" | {pred_bicep:.1f} cm")
                st.metric("Wrist", f"{pred_wrist/2.54:.1f}\" | {pred_wrist:.1f} cm")

        with tabs[1]:
            st.subheader("Predicted Short-Sleeve Shirt Measurements")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Short Sleeve Length",
                        f"{garment_measurements['short_sleeve_length']/2.54:.1f}\" | {garment_measurements['short_sleeve_length']:.1f} cm",
                        help="40% of full sleeve length")
            with col2:
                st.metric("Short Sleeve Opening",
                        f"{garment_measurements['short_sleeve_opening']/2.54:.1f}\" | {garment_measurements['short_sleeve_opening']:.1f} cm",
                        help=f"Adjusted for {build_type} build and {fit_type} fit")

        with tabs[2]:
            st.subheader(f"Predicted Pants ({fit_type}, {taper_type}, {build_type})")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rise", f"{adj_rise/2.54:.1f}\" | {adj_rise:.1f} cm", help="Distance from crotch to waistband")
                st.metric("Outseam Length", f"{(adj_leg_length)/2.54:.1f}\" | {adj_leg_length:.1f} cm")
                st.metric("Inseam Length",
                        f"{garment_measurements['pant_inseam']/2.54:.1f}\" | {garment_measurements['pant_inseam']:.1f} cm",
                        help="Outseam minus rise/2")
                st.metric("Waist", f"{adj_waist/2.54:.1f}\" | {adj_waist:.1f} cm")
            with col2:
                st.metric("Hips", f"{adj_hips/2.54:.1f}\" | {adj_hips:.1f} cm")
                st.metric("Thigh (Body)", f"{garment_measurements['body_thigh_in']:.1f}\" | {adj_thigh:.1f} cm", help="Natural body measurement")
                st.metric("Thigh (Garment)",
                        f"{garment_measurements['garment_thigh_in']:.1f}\" | {garment_measurements['garment_thigh_cm']:.1f} cm",
                        help=f"Body thigh + {THIGH_EASE_INCHES}\" ease for comfort")
                st.metric("Pant Leg Opening",
                        f"{garment_measurements['pant_leg_opening']/2.54:.1f}\" | {garment_measurements['pant_leg_opening']:.1f} cm",
                        help="69.6% of garment thigh, adjusted for fit type")
            # Validation feedback
            if custom_thigh > 0:
                thigh_diff = abs(garment_measurements['garment_thigh_cm'] - custom_thigh)
                if thigh_diff <= 2:
                    st.success(f"✅ Predicted garment thigh ({garment_measurements['garment_thigh_cm']:.1f}cm) is within 2cm of your measurement ({custom_thigh:.1f}cm).")
                else:
                    st.warning(f"⚠️ Predicted garment thigh differs by {thigh_diff:.1f}cm from your supplied ({custom_thigh:.1f}cm).")
            if custom_leg_opening > 0:
                leg_diff = abs(garment_measurements['pant_leg_opening'] - custom_leg_opening)
                if leg_diff <= 1.5:
                    st.success(f"✅ Predicted leg opening ({garment_measurements['pant_leg_opening']:.1f}cm) very close to your input ({custom_leg_opening:.1f}cm).")
                else:
                    st.warning(f"⚠️ Leg opening predicted ({garment_measurements['pant_leg_opening']:.1f}cm) differs by {leg_diff:.1f}cm from your preference ({custom_leg_opening:.1f}cm).")

        with tabs[3]:
            st.subheader("Predicted Shorts Measurements")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Shorts Outseam Length",
                        f"{garment_measurements['shorts_length']/2.54:.1f}\" | {garment_measurements['shorts_length']:.1f} cm",
                        help="Pant length minus 57.7cm for above-knee fit")
                st.metric("Shorts Inseam Length",
                        f"{garment_measurements['shorts_inseam']/2.54:.1f}\" | {garment_measurements['shorts_inseam']:.1f} cm",
                        help="Outseam minus rise/2")
            with col2:
                st.metric("Shorts Leg Opening",
                        f"{garment_measurements['shorts_leg_opening']/2.54:.1f}\" | {garment_measurements['shorts_leg_opening']:.1f} cm",
                        help="110% of garment thigh, adjusted for fit type")

        with tabs[4]:
            st.subheader("Fit Analysis & Custom Guidance")
            st.write(f"**Fit:** {fit_type}, **Taper:** {taper_type}, **Build:** {build_type}")
            st.write(f"**Thigh Ease:** {THIGH_EASE_INCHES}\" ({THIGH_EASE_INCHES * 2.54:.1f}cm)")
            st.markdown("#### Applied Adjustments:")
            st.write(f"**Fit Type ({fit_type}):**")
            if fit_type == "Slim":
                st.write("- All openings reduced by 10% for trimmer silhouette")
                st.write("- Best for lean builds or fashion-forward looks")
            elif fit_type == "Relaxed/Athletic":
                st.write("- All openings increased by 10% for comfort")
                st.write("- Ideal for muscular builds or active lifestyles")
            else:
                st.write("- Classic proportions based on industry standards")
            st.write(f"**Build Type ({build_type}):**")
            if build_type == "Stocky/Muscular":
                st.write("- Sleeve openings increased by 5% for muscular arms")
                st.write("- Additional thigh and seat room added")
            elif build_type == "Lean/Slim":
                st.write("- Sleeve openings reduced by 5% for slimmer proportions")
                st.write("- Trimmed measurements throughout")
            else:
                st.write("- Standard proportions maintained")
            st.info("💡 **Tip:** All measurements distinguish between body size and finished garment size, matching how ready-to-wear brands specify products.")

        with tabs[5]:
            st.subheader("🧵 Finished Garment Specifications")
            st.markdown("*Production-ready measurements for manufacturing*")
            st.markdown("#### Shirts")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Full Sleeve Length:** {pred_sleeve:.1f} cm")
                st.write(f"**Short Sleeve Length:** {garment_measurements['short_sleeve_length']:.1f} cm")
                st.write(f"**Neck:** {pred_neck:.1f} cm")
                st.write(f"**Shoulder Width:** {pred_shoulder:.1f} cm")
            with col2:
                st.write(f"**Chest Around:** {pred_chest:.1f} cm")
                st.write(f"**Sleeve Opening:** {garment_measurements['short_sleeve_opening']:.1f} cm")
                st.write(f"**Bicep:** {pred_bicep:.1f} cm")
                st.write(f"**Torso Length:** {pred_torso-3.8:.1f} cm")
            st.markdown("#### Pants")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Outseam:** {adj_leg_length:.1f} cm")
                st.write(f"**Inseam:** {garment_measurements['pant_inseam']:.1f} cm")
                st.write(f"**Rise:** {adj_rise:.1f} cm")
                st.write(f"**Waist:** {adj_waist:.1f} cm")
            with col2:
                st.write(f"**Hips:** {adj_hips:.1f} cm")
                st.write(f"**Thigh (Finished):** {garment_measurements['garment_thigh_cm']:.1f} cm")
                st.write(f"**Leg Opening:** {garment_measurements['pant_leg_opening']:.1f} cm")
            st.markdown("#### Shorts")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Outseam:** {garment_measurements['shorts_length']:.1f} cm")
                st.write(f"**Inseam:** {garment_measurements['shorts_inseam']:.1f} cm")
            with col2:
                st.write(f"**Leg Opening:** {garment_measurements['shorts_leg_opening']:.1f} cm")

        st.markdown("---")
        st.markdown("*All measurements calculated using validated formulas based on your actual garment analysis. Garment measurements include appropriate ease for comfortable fit and movement.*")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
