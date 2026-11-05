

import streamlit as st
import joblib

st.set_page_config(
    page_title="NLP Customer Support",
    page_icon="💬",
    layout="wide"
)

# Load trained NLP model
model = joblib.load("customer_support_model.pkl")

# Load response mapping
response_mapping = joblib.load("response_mapping.pkl")

st.title("💬 NLP Customer Support System")

st.write(
    "AI-powered customer inquiry classification "
    "and automated response suggestions."
)

page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Customer Support"]
)

if page == "Home":

    st.header("Customer Support Dashboard")

    st.info(
        "Our NLP model automatically classifies customer "
        "inquiries and suggests responses."
    )

elif page == "Customer Support":

    st.header("Customer Inquiry")

    message = st.text_area(
        "Enter your customer message",
        placeholder="Example: I have not received my refund"
    )

    if st.button("Submit Inquiry"):

        if message.strip():

            # Predict customer inquiry category
            category = model.predict([message])[0]

            # Calculate prediction confidence
            probabilities = model.predict_proba([message])[0]

            confidence = max(probabilities) * 100

            # Get suggested response
            response = response_mapping.get(
                category,
                "Thank you for contacting customer support. "
                "Our team will assist you shortly."
            )

            st.success("Inquiry analyzed successfully!")

            st.subheader("Predicted Category")

            st.write(category)

            st.subheader("Prediction Confidence")

            st.write(f"{confidence:.2f}%")

            st.subheader("Suggested Customer Response")

            st.info(response)

        else:

            st.warning("Please enter a customer inquiry.")
