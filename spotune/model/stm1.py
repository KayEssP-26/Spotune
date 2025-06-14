import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import model
SPOTIPY_CLIENT_ID = 'CLIENT ID'
SPOTIPY_CLIENT_SECRET = 'CLIENT SECRET'

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

def listfix(str_list):
    formatted_list = eval(str_list)
    return ', '.join(formatted_list)

song_name_list = pd.read_csv('newdata_songlist.csv')
song_dataset = pd.read_csv('clean_newdataset.csv')

st.title("Music Recommendation System")
st.header("Search for a song and choose the number of recommendations...")

search_query = st.text_input("Search for a song:", placeholder="Enter song name...")

if search_query:
    matching_songs = song_name_list[song_name_list['track_info'].str.contains(search_query, case=False, na=False)]
    if not matching_songs.empty:
        song_select = st.selectbox("Select a song from the results:", matching_songs['track_info'].tolist())
        st.write("You selected:", song_select)

        songindex = matching_songs[matching_songs['track_info'] == song_select].index[0]
        song_name = song_dataset.loc[songindex, 'track_name']
    else:
        st.error("No matching song found. Please try again.")
else:
    st.info("Please enter a song name to search.")
    song_select = None

song_count = st.slider('Number of Recommendations', 1, 10, 5)

def get_spotify_details(spotify_uri):
    track_info = sp.track(spotify_uri)
    song_cover_art = track_info['album']['images'][0]['url']
    spotify_link = track_info['external_urls']['spotify']
    return song_cover_art, spotify_link

if st.button('Get Recommendations') and song_select:
    recommendations = model.find_similar_songs(song_name=song_name, num_recommendations=song_count)

    for index, row in recommendations.iterrows():
        with st.container():
            cols = st.columns([3, 3, 5, 2])

            spotify_uri = row['track_id'] 
            song_cover_art, spotify_link = get_spotify_details(spotify_uri)

            with cols[0]:
                st.image(song_cover_art, use_column_width=True)

            with cols[1]:
                st.markdown(f"[**{row['track_name']}**]({spotify_link})")
                st.write(row['artists'])

            with cols[2]:
                st.write(f"Album: {row['album_name']}")

            with cols[3]:
                st.write(f"Genre: {row['track_genre']}")

            st.write("---")

st.write("Enjoy your music!")
