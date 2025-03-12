import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.image("2.png", width=200)
st.title("👔 AI-Powered Body Measurement Predictor - MEN")

# Function to load the dataset
@st.cache_data
def load_data():
    file_path = 'Sizing Spreadsheet - Test Data - 13.05.2024.csv'
    df = pd.read_csv(file_path)
    return df

# Load the dataset
df = load_data()

# Function to train the model
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
    rmse = mse ** 0.5  # calculate RMSE manually
    r2 = r2_score(y_test, y_pred)
    return model, mae, mse, rmse, r2

# Train the model
model, mae, mse, rmse, r2 = train_model(df)
st.subheader("Model Evaluation Metrics")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Root Mean Squared Error: {rmse:.2f}")
st.write(f"R-squared: {r2:.2f}")

st.subheader("OPEN SIDEBAR -button at top left- to input values.")

# Sidebar inputs
st.sidebar.header("Enter Your Details")
name = st.sidebar.text_input("Name (Optional)")
email = st.sidebar.text_input("Email (Optional)")

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

if st.sidebar.button("Predict Measurements"):
    new_data = pd.DataFrame({
        'Weight': [weight],
        'Height': [height],
        'Chest': [chest],
        'Abdomen': [abdomen]
    })
    predicted_values = model.predict(new_data)
    
    pred_values = {
        "Neck (CM)": round(predicted_values[0, 0], 1),
        "Sleeve Length (CM)": round(predicted_values[0, 1], 1),
        "Shoulder Width (CM)": round(predicted_values[0, 2], 1),
        "Chest Around (CM)": round(predicted_values[0, 3], 1),
        "Stomach (CM)": round(predicted_values[0, 4], 1),
        "Torso Length (CM)": round(predicted_values[0, 5] - 3.81, 1),
        "Bicep (CM)": round(predicted_values[0, 6], 1),
        "Wrist (CM)": round(predicted_values[0, 7], 1),
        "Rise (CM)": round(predicted_values[0, 8], 1),
        "Leg Length (CM)": round(predicted_values[0, 9] - 3.8, 1),
        "Waist (CM)": round(predicted_values[0, 10], 1),
        "Hips (CM)": round(predicted_values[0, 11], 1),
        "Thigh (CM)": round(predicted_values[0, 12], 1),
    }

    st.subheader("Predicted Measurements")
    edited_values = st.data_editor(pd.DataFrame([pred_values]))

    if st.button("Submit"):
        customer_data = {
            "Name": name,
            "Email": email,
            "Weight": weight,
            "Height": height,
            "Chest": chest,
            "Abdomen": abdomen,
        }
        combined_data = {**customer_data, **edited_values.iloc[0].to_dict()}
        df_to_save = pd.DataFrame([combined_data])

        # Save to CSV (append mode)
        try:
            existing_df = pd.read_csv("customer_measurements.csv")
            updated_df = pd.concat([existing_df, df_to_save], ignore_index=True)
        except FileNotFoundError:
            updated_df = df_to_save

        updated_df.to_csv("customer_measurements.csv", index=False)

        st.success("Your measurements have been submitted successfully!")
