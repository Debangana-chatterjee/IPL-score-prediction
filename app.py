import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import pandas as pd

def load_lstm_model():
    return load_model("lstm_model.h5")

model = load_lstm_model()

def load_encoder():
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return encoder

def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

encoder = load_encoder()
scaler = load_scaler()

st.title("🏏 IPL Score Prediction App")
teams = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore", "Kolkata Knight Riders", "Sunrisers Hyderabad", "Delhi Daredevils", "Pune Warriors", "Rajasthan Royals", "Gujarat Lions","Rising Pune Supergiant","Kings XI Punjab","Deccan Chargers","Kochi Tuskers Kerala","Rising Pune Supergiants"]
venues = ["Wankhede Stadium", "M Chinnaswamy Stadium", "Eden Gardens","MA Chidambaram Stadium, Chepauk", "Rajiv Gandhi International Stadium, Uppal","Punjab Cricket Association Stadium, Mohali","Feroz Shah Kotla","Sawai Mansingh Stadium","Dr DY Patil Sports Academy","Newlands","St George's Park","Kingsmead","SuperSport Park","New Wanderers Stadium","Brabourne Stadium","Sardar Patel Stadium, Motera","Barabati Stadium","Vidarbha Cricket Association Stadium, Jamtha","Himachal Pradesh Cricket Association Stadium","Nehru Stadium","Holkar Cricket Stadium","Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium","Shaheed Veer Narayan Singh International Stadium","JSCA International Stadium Complex","Sheikh Zayed Stadium","Sharjah Cricket Stadium","Dubai International Cricket Stadium","Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium","Maharashtra Cricket Association Stadium","Saurashtra Cricket Association Stadium","Green Park"]

bowl_teams=teams.copy()

team = st.selectbox("Select Batting Team",teams )
bowl_team = st.selectbox("Select Bowling Team", bowl_teams)
venue = st.selectbox("Select Venue",venues)
over = st.number_input("Enter Over Number", min_value=1, max_value=20, step=1)
wickets = st.number_input("Enter Wickets Fallen", min_value=0, max_value=10, step=1)
runs_last_5 = st.number_input("Enter Runs Scored in Last 5 Overs", min_value=0, step=1)



if st.button("Predict Score"):
    input_data = pd.DataFrame([[team,bowl_team, venue, over, wickets, runs_last_5]],
                            columns=["bat_team","bowl_team","venue", "overs", "wickets", "runs_last_5"])

    categorical_features = input_data[["venue", "bat_team","bowl_team"]]
    encoded_features = encoder.transform(categorical_features)
    try:
        encoded_features = encoded_features.toarray()
    except AttributeError:
        pass

    numerical_features = input_data[["wickets","overs","runs_last_5"]].values
    scaled_features = scaler.transform(numerical_features)

    full_features = np.concatenate([encoded_features, scaled_features], axis=1)
    lstm_input = full_features.reshape((1, 1, full_features.shape[1]))

    prediction = model.predict(lstm_input)
    st.write(f"Predicted Final Score: {prediction[0][0]:.2f}")
