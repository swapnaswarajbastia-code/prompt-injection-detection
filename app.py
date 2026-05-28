import streamlit as st
import pickle
import os

# Load model
try:
    sgd_model = pickle.load(open("sgd_model.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Error: sgd_model.pkl not found. Make sure the file is in the repository.")
    st.stop()

# Load vectorizer
try:
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Error: vectorizer.pkl not found. Make sure the file is in the repository.")
    st.stop()

# Title
st.title("Prompt Injection Detection")

# Input
user_input_text = st.text_area("Enter text")

# Prediction
if st.button("Predict"):

    user_input_vectorized = vectorizer.transform([user_input_text])

    predicted_category = sgd_model.predict(user_input_vectorized)

    st.success(f"Prediction: {predicted_category[0]}")