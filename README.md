
# 🎵 Music Recommendation System using Neural Networks

This project is a neural network-based music recommendation system that leverages audio features such as **tempo**, **acousticness**, **danceability**, **valence**, and more to suggest similar songs. The system is built using **TensorFlow**, **Scikit-learn**, and a **Streamlit** frontend integrated with the **Spotify API** for rich music metadata and visuals.

---

## 🚀 Features

- Recommend similar songs based on audio features using deep learning
- Cleaned and scaled feature set from a rich audio dataset
- Streamlit web interface for real-time interaction
- Spotify API integration for song details and album art
- Dynamic user search and recommendation slider

---

## 🔍 Dataset

The dataset contains audio features extracted from songs, including:

- `tempo`, `acousticness`, `danceability`, `energy`, `valence`
- `popularity`, `duration_ms`, `explicit`, `key`, `loudness`, `mode`
- `speechiness`, `instrumentalness`, `liveness`, `time_signature`
- Metadata: `track_name`, `artists`, `album_name`, `track_id`, `track_genre`

---

## 🧠 Model Architecture

Implemented in `model.py` using Keras:

- Input Layer: 15 audio features
- Hidden Layers:
  - Dense (128 units, ReLU) + Dropout (0.2)
  - Dense (64 units, ReLU) + Dropout (0.2)
- Output Layer: Linear activation with 15 outputs (same shape as input)
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam (learning rate = 0.01)
- Evaluation Metrics: MAE, MSE, R²

---

## 🎯 How it Works

1. User searches for a song name.
2. The system finds the matching song in the dataset.
3. A trained neural network predicts transformed features.
4. Euclidean distance is used to find the nearest songs in feature space.
5. Top N similar songs are returned with Spotify details.

---
