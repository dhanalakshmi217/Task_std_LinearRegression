import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load model and scaler
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Student Performance Prediction")

st.title("🎓 Student Performance Prediction App")
st.write("Enter student details to predict final score")

# -------------------------------
# INPUTS with LIMITS
# -------------------------------
study_hours = st.number_input(
    "📘 Study Hours (1 to 50)",
    min_value=1,
    max_value=50,
    value=5
)

attendance = st.number_input(
    "🏫 Attendance Percentage (1 to 100)",
    min_value=1,
    max_value=100,
    value=75
)

previous_score = st.number_input(
    "📝 Previous Exam Score (1 to 100)",
    min_value=1,
    max_value=100,
    value=60
)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):
    try:
        # Prepare input
        input_data = np.array([[study_hours, attendance, previous_score]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        st.success(f"✅ Predicted Final Score: {round(prediction[0], 2)}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
