import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

file_path = "clean_newdataset.csv"
data = pd.read_csv(file_path)
df = pd.read_csv(file_path)

features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
       'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X = df[features].values
y = df[features].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(len(features),)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(features), activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.2)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

def find_similar_songs(song_name, num_recommendations):
    # Find the song in the dataset
    song = df[df['track_name'] == song_name]
    if song.empty:
        return f"Song '{song_name}' not found in the dataset."

    song_features = song[features].iloc[0].values.reshape(1, -1)

    all_features = model.predict(X)

    distances = np.linalg.norm(all_features - song_features, axis=1)

    top_indices = np.argsort(distances)[:num_recommendations]

    recommendations = data.iloc[top_indices][['track_id', 'artists', 'album_name', 'track_name', 'track_genre']]
    
    return recommendations
