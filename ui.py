import streamlit as st
import pandas as pd
import requests
import json

@st.cache_data
def get_categoriess():
    with open("model/categories.json", "r") as f:
        return json.load(f)
cats = get_categoriess()

st.title("🎧 Podcast Listening Time Predictor")

col1, col2 = st.columns(2)

with col1:
    podcast_name = st.selectbox("Podcast Name", cats['podcasts'])
    episode_title = st.selectbox("Episode Title", cats['titles']) # Changed to selectbox
    genre = st.selectbox("Genre", cats['genres'])
    pub_day = st.selectbox("Publication Day", cats['days'])
    pub_time = st.time_input("Publication Time")

with col2:
    ep_length = st.number_input("Episode Length (mins)", min_value=1, value=45)
    host_pop = st.slider("Host Popularity (%)", 0, 100, 50)
    guest_pop = st.slider("Guest Popularity (%)", 0, 100, 50)
    num_ads = st.number_input("Number of Ads", min_value=0, value=2)
    sentiment = st.slider("Sentiment (0=Neg, 1=Pos)", 0.0, 1.0, 0.5)

if st.button("Predict"):
    payload = {
        "Podcast_Name": podcast_name,
        "Episode_Title": episode_title,
        "Episode_Length_minutes": ep_length,
        "Genre": genre,
        "Host_Popularity_percentage": host_pop,
        "Publication_Day": pub_day,
        "Publication_Time": str(pub_time),
        "Guest_Popularity_percentage": guest_pop,
        "Number_of_Ads": num_ads,
        "Episode_Sentiment": sentiment
    }
    
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload)
        st.metric("Predicted Listening Time", f"{response.json()['listening_time_minutes']} mins")
    except Exception as e:
        st.error(f"Connection Error: {e}")