import streamlit as st
import pickle

# Load saved model
sgd_model = pickle.load(open("sgd_model.pkl", "rb"))

# Load TF-IDF vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page title
st.title("Prompt Injection Detection System")

# Description
st.write("This app predicts the category of prompt injection attacks using SGD Classifier.")

# User input box
user_input_text = st.text_area("Enter text for prediction")

# Prediction button
if st.button("Predict"):

    # Check empty input
    if user_input_text.strip() != "":

        # Convert text into numerical form
        user_input_vectorized = vectorizer.transform([user_input_text])

        # Predict category
        predicted_category = sgd_model.predict(user_input_vectorized)

        # Show prediction
        st.success(f"Predicted Category: {predicted_category[0]}")

    else:
        st.warning("Please enter some text")