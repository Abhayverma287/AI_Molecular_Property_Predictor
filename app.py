import streamlit as st

from src.predict import predict_property

st.title("AI Molecular Property Prediction Platform")

smiles = st.text_input("Enter Molecule SMILES")

if st.button("Predict Property"):

    result = predict_property(smiles)

    st.success(result)