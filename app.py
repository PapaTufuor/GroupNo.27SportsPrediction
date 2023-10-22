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


feature_name_mapping ={
    "Value in Euros":top_features_list[0],
    "Age":top_features_list[1],
    "Release Clause": top_features_list[2],
    "Movement Reactions": top_features_list[3],
    "Potential": top_features_list[4],
    "Goalkeeping": top_features_list[5],
    "Wages": top_features_list[6]
}

st.title('FIFA Player Rating Predictor')
st.sidebar.header('Enter Feature Values')





user_inputs={}
for user_friendly_name, actual_name in feature_name_mapping.items():
    user_inputs[user_friendly_name] = st.sidebar.number_input(f'Enter value for {user_friendly_name}', value=0.0)

if st.sidebar.button('Predict'):
    input_data=pd.DataFrame({actual_name: user_inputs[user_friendly_name] for user_friendly_name, actual_name in feature_name_mapping.items()}, index=[0])

    scaled_input_data=scaler.transform(input_data)

    predicted_rating = model.predict(scaled_input_data)

    


    max_possible_rating=100.0

    deviation=abs(predicted_rating[0] - max_possible_rating)
    confidence=1.0 - (deviation/max_possible_rating)

    

    st.write(f'Predicted Rating: {predicted_rating[0]:.2f}')
    st.write(f'Confidence Level: {confidence * 100:.2f}%')

