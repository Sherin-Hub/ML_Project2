import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction")
st.info("Enter house details and click **Predict Price**")

# Load dataset (KEEP CSV IN SAME FOLDER AS app.py)
@st.cache_data
def load_data():
    return pd.read_csv("house_data.csv")

data = load_data()

# Features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Multiple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Input section
st.subheader("🔢 Enter House Details")

area = st.number_input("📐 Area (sq.ft)", min_value=300, max_value=5000, value=1200)
bedrooms = st.slider("🛏️ Number of Bedrooms", 1, 10, 3)
location = st.selectbox(
    "📍 Location Rating",
    ["Poor", "Average", "Good", "Very Good", "Excellent"]
)

# Encode location
location_map = {
    "Poor": 1,
    "Average": 2,
    "Good": 3,
    "Very Good": 4,
    "Excellent": 5
}
location_encoded = location_map[location]

# Prepare input
input_data = pd.DataFrame([[
    area,
    bedrooms,
    location_encoded
]], columns=X.columns)

input_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔮 Predict House Price"):
    prediction = model.predict(input_scaled)[0]

    st.subheader("📊 Prediction Result")
    st.metric("Estimated House Price", f"₹ {round(prediction, 2)} Lakhs")

    avg_price = y.mean()
    diff = ((prediction - avg_price) / avg_price) * 100

    st.write(f"**Average Market Price:** ₹ {round(avg_price, 2)} Lakhs")
    st.write(f"**Price Difference:** {diff:.2f}%")

    if diff > 20:
        st.warning("📈 High Price Area")
    elif diff < -10:
        st.info("📉 Affordable Area")
    else:
        st.success("✅ Fair Market Price")
