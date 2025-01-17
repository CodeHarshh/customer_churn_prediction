import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Load pre-trained encoders and scaler
with open('Label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('OneHot_Encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title('Customer Churn Prediction')

# Collecting user inputs
credit_score = st.number_input('Credit Score')
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', min_value=18, max_value=92)
tenure = st.slider('Tenure (years)', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.number_input('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary')
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [1 if gender == 'Male' else 0],  # Encode gender
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]
})

# One-hot encode the 'geography' feature
geo_encoded = one_hot_encoder.transform([[geography]])
geo_encoder_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))
print(geo_encoder_df)

# Append the one-hot encoded geography features to input_data
input_data = pd.concat([input_data, geo_encoder_df], axis=1)

# Drop the original 'geography' column
input_data.drop(columns=['Geography'], inplace=True)


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn probability
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]


# Display the results
st.write(f'Churn Probability: {prediction_proba:.2f}')
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
