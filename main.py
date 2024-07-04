import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import  StandardScaler

import numpy as np

model_path = 'models/RandomForest.joblib'

try:
    loaded_model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found.")
    loaded_model = None
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None


def predictions(loaded_model,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    if loaded_model is None:
        return "Model not loaded"
     
    input_data = {
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    }
    input_df = pd.DataFrame(input_data)

    # Initialize StandardScaler and LabelEncoder
    ss = StandardScaler()
    for col in input_df:
        input_df[col] = ss.fit_transform(input_df[[col]])

    features = input_df.values.tolist()[0]
    predicted_price = loaded_model.predict([features])

    print(predicted_price[0])
    return predicted_price[0]


def main():
    st.title('Welcome Diabetes Check system')
    st.header("Please enter your details to proceed your result")


    Pregnancies= st.number_input("Pregnancies", min_value=0, max_value=17)
    Glucose= st.number_input("Glucose", min_value=0.0, max_value=199.0)
    BP= st.number_input("BloodPressure", min_value=0.0, max_value=122.0)
    SkinThickness= st.number_input("SkinThickness", min_value=0.0, max_value=99.0)
    Insulin= st.number_input("Insulin", min_value=0.0, max_value=846.0)
    BMI= st.number_input("BMI", min_value=0.0, max_value=67.1)
    DiabetesPedigreeFunction= st.number_input(" DiabetesPedigreeFunction", min_value=0.0, max_value=2.42)
    Age=st.number_input("Age", min_value=0, max_value=80,step=1)
    

    if st.button('Predict'):
        result = predictions(loaded_model, Pregnancies,Glucose,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        if result=='0':
            st.success(f"Patient does not have diabetes")
        else:    
            st.warning(f"Patient has diabetes")

if __name__=='__main__':
    main()   