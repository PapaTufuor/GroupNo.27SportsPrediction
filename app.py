import streamlit as st
import pandas as pd
import pickle
import joblib

#loading the scaler
scaler=joblib.load('C:\\Users\\hp\\Desktop\\Mid Sem Project\\scaler.pkl')


#loading the Random Forest Regressor model
model=joblib.load('C:\\Users\\hp\\Desktop\\Mid Sem Project\\model.pkl')

#loading the list of features used to train the model
with open('C:\\Users\\hp\\Desktop\\Mid Sem Project\\top_features_list.pkl','rb') as file:
    top_features_list=pickle.load(file)

# print(type(model))



st.title('FIFA Player Rating Predictor')
st.sidebar.header('Enter Feature Values')

user_inputs={}
for feature in top_features_list:
    user_inputs[feature] = st.sidebar.number_input(f'Enter value for {feature}', value=0.0)

if st.sidebar.button('Predict'):
    input_data=pd.DataFrame(user_inputs, index=[0])

    scaled_input_data=scaler.transform(input_data)

    predicted_rating = model.predict(scaled_input_data)

    


    max_possible_rating=100.0

    if predicted_rating[0] ==max_possible_rating:
        confidence=1.0
    else:
        confidence=1.0 -abs(predicted_rating[0] - max_possible_rating)/max_possible_rating

    st.write(f'Predicted Rating: {predicted_rating[0]:.2f}')
    st.write(f'Confidence Level: {confidence * 100:.2f}%')

