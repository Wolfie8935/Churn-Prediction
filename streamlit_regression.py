import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
import pickle

model = tf.keras.models.load_model('./regression_model.h5')


with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('one_encoder.pkl','rb') as file:
    one_encoder = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
    

st.title('Estimated Salary Prediction')


geography = st.selectbox('Geography', one_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

geo_encoded = one_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

pred = model.predict(input_data_scaled)
pred_sal = pred[0][0]

st.write(f'Predicted Estimated Salary: ${pred_sal:.2f}')