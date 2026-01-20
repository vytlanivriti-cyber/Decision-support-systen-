import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("📺 YouTube Video Performance Predictor")
st.write("A Decision Support System for predicting video virality.")

# User Inputs
likes = st.number_input("Likes", min_value=0)
comments = st.number_input("Comments", min_value=0)
duration = st.number_input("Duration (seconds)", min_value=0)
title = st.text_input("Video Title")

# Feature Engineering
if st.button("Predict"):
    title_length = len(title)
    word_count = len(title.split())

    log_likes = np.log1p(likes)
    log_comments = np.log1p(comments)

    input_data = pd.DataFrame({
        "log_likes": [log_likes],
        "log_comments": [log_comments],
        "duration_min": [duration / 60],
        "word_count": [word_count]
    })

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🔥 This video is likely to go VIRAL! (Probability: {prob:.2f})")
    else:
        st.error(f"📉 This video may NOT go viral. (Probability: {prob:.2f})")
