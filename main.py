import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Load the trained model
with open('final_xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Titanic Survival Prediction')
st.write('Enter passenger details to predict survival:')

# Input fields for passenger features
pclass = st.selectbox('Pclass (Ticket Class)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 80, 28) # Default to median age
sibsp = st.slider('SibSp (Number of Siblings/Spouses Aboard)', 0, 8, 0)
fare = st.number_input('Fare', min_value=0.0, max_value=500.0, value=30.0)
embarked = st.selectbox('Embarked (Port of Embarkation)', ['C', 'Q', 'S']) # Cherbourg, Queenstown, Southampton

# Preprocess inputs (similar to how the training data was preprocessed)
# Convert 'Sex' to numerical
sex_encoded = 1 if sex == 'male' else 0

# Convert 'Embarked' to numerical
embarked_map = {'C': 0, 'Q': 1, 'S': 2}
embarked_encoded = embarked_map.get(embarked, 2) # Default to 'S' if unknown

# Create a DataFrame for prediction
input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, fare, embarked_encoded]],
                          columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked'])

# Make prediction
if st.button('Predict Survival'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f'The passenger is likely to survive! (Probability: {prediction_proba[0][1]:.2f})')
    else:
        st.error(f'The passenger is likely not to survive. (Probability: {prediction_proba[0][0]:.2f})')

st.write("\n--- Notes ---")
st.write("Pclass: 1 = 1st, 2 = 2nd, 3 = 3rd")
st.write("SibSp: Number of siblings/spouses aboard the Titanic")
st.write("Embarked: C = Cherbourg, Q = Queenstown, S = Southampton")
