import subprocess
import sys
import joblib
import os

# Explicitly install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn==0.24.2", "joblib==1.0.1"])


import streamlit as st
import pandas as pd
import pickle

# Load the trained model
#model = joblib.load('diabetes_model.joblib')
model = joblib.load('diabetes_test.pkl')

# Streamlit app description
st.write("""
# Diabetes Prediction App
This app predicts whether a person has diabetes based on some features. Please input the following values to assess the diabetes status.
""")

# Input features with explanations
st.header('User Input Features')

# User inputs with explanations
def user_input_features():
    st.write('Number of times pregnant')
    pregnancies = st.slider('Pregnancies', 0, 17, 3)

    st.write('Plasma glucose concentration')
    glucose = st.slider('Glucose', 0, 200, 100)

    st.write('Diastolic blood pressure (mm Hg)')
    blood_pressure = st.slider('Blood Pressure', 0, 122, 70)

    st.write('Triceps skin fold thickness (mm)')
    skin_thickness = st.slider('Skin Thickness', 0, 99, 20)

    st.write('2-Hour serum insulin (mu U/ml)')
    insulin = st.slider('Insulin', 0, 846, 79)

    st.write('Body mass index (weight in kg/(height in m)^2)')
    bmi = st.slider('BMI', 0.0, 67.1, 31.4)

    st.write('Diabetes pedigree function')
    diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)

    st.write('Age (years)')
    age = st.slider('Age', 21, 81, 29)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction button
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    if prediction[0] == 0:
        st.success('Great news! The assessment suggests that you are unlikely to have diabetes. Keep up the healthy lifestyle!')
    else:
        st.error('Hey there! The assessment suggests that you might have diabetes. We recommend consulting a healthcare professional for further guidance.')
