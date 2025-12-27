import streamlit as st
import os
import pandas as pd
import requests
import json
import logging
from typing import Dict, Any

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/streamlit_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@st.cache_data
def get_categories() -> Dict[str, list]:
    """Loads category data for the UI selectors."""
    try:
        with open("model/categories.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading categories: {e}")
        return {"podcasts": [], "titles": [], "genres": [], "days": []}

cats: Dict[str, list] = get_categories()

col1, col2 = st.columns(2)

with col1:
    podcast_name: str = st.selectbox("Podcast Name", cats.get('podcasts', []))
    episode_title: str = st.selectbox("Episode Title", cats.get('titles', []))
    genre: str = st.selectbox("Genre", cats.get('genres', []))
    pub_day: str = st.selectbox("Publication Day", cats.get('days', []))
    pub_time = st.time_input("Publication Time")

with col2:
    ep_length: int = st.number_input("Episode Length (mins)", min_value=1, value=45)
    host_pop: int = st.slider("Host Popularity (%)", 0, 100, 50)
    guest_pop: int = st.slider("Guest Popularity (%)", 0, 100, 50)
    num_ads: int = st.number_input("Number of Ads", min_value=0, value=2)
    sentiment: float = st.slider("Sentiment (0=Neg, 1=Pos)", 0.0, 1.0, 0.5)

if st.button("Predict"):
    payload: Dict[str, Any] = {
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
        logger.info(f"Request for: {podcast_name}")
        response: requests.Response = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=10)
        response.raise_for_status()
        
        prediction: float = response.json().get('listening_time_minutes', 0.0)
        st.metric("Predicted Listening Time", f"{prediction:.2f} mins")
        logger.info("Prediction successful.")
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        logger.error(f"UI prediction error: {e}")