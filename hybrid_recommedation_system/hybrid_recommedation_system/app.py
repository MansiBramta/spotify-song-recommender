from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz

from content_based_filtering import content_recommendation
from collaborative_filtering import collaborative_recommendation


# -------------------------------------------------- Page configuration

st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
)


# -------------------------------------------------- Data directory

DATA = Path(__file__).parent.parent / "data"


# -------------------------------------------------- Load data

@st.cache_resource(show_spinner="Loading data...")
def load_data():
    return {
        "songs_data": pd.read_csv(DATA / "cleaned_data.csv"),
        "transformed_data": load_npz(DATA / "transformed_data.npz"),
        "track_ids": np.load(DATA / "track_ids.npy", allow_pickle=True),
        "filtered_data": pd.read_csv(DATA / "collab_filtered_data.csv"),
        "interaction_matrix": load_npz(DATA / "interaction_matrix.npz"),
    }


# Check whether the required data files exist

required_files = [
    "cleaned_data.csv",
    "transformed_data.npz",
    "track_ids.npy",
    "collab_filtered_data.csv",
    "interaction_matrix.npz",
]

missing_files = [
    filename
    for filename in required_files
    if not (DATA / filename).exists()
]

if missing_files:
    st.error(
        "Required data files are missing:\n\n"
        + "\n".join(f"- {filename}" for filename in missing_files)
        + "\n\nRun `python build_data.py` from your project folder first."
    )
    st.stop()


# Load the data

d = load_data()

songs_data = d["songs_data"]
transformed_data = d["transformed_data"]
filtered_data = d["filtered_data"]
track_ids = d["track_ids"]
interaction_matrix = d["interaction_matrix"]


# -------------------------------------------------- Display recommendations

def show_recommendations(recs):
    if recs.empty:
        st.warning("No recommendations could be generated for this song.")
        return

    for ind, (_, rec) in enumerate(recs.iterrows()):

        title = (
            f"**{str(rec['name']).title()}** by "
            f"**{str(rec['artist']).title()}**"
        )

        if ind == 0:
            st.markdown("## First Recommendation")
        else:
            st.markdown(f"### {ind + 1}. Next Up 🎵")

        st.markdown(f"#### {title}")

        url = rec.get("spotify_preview_url")

        if isinstance(url, str) and url.strip():
            st.audio(url)
        else:
            st.caption("No preview available")

        st.write("---")


# -------------------------------------------------- Streamlit interface

st.title("🎵 Spotify Song Recommender")

st.write(
    "Enter a song and artist, then choose how you want "
    "to discover your next songs."
)

song_name = st.text_input("Enter a song name:").strip().lower()
artist_name = st.text_input("Enter the artist name:").strip().lower()

k = st.selectbox(
    "How many recommendations do you want?",
    [5, 10, 15, 20],
    index=1,
)

filtering_type = st.radio(
    "Choose a recommendation method:",
    [
        "Content-Based Filtering",
        "Collaborative Filtering",
    ],
)


# -------------------------------------------------- Generate recommendations

if st.button("Get Recommendations", type="primary"):

    if not song_name or not artist_name:
        st.warning("Please enter both the song name and artist name.")

    elif filtering_type == "Content-Based Filtering":

        # Check whether the song exists in the content dataset
        song_names = songs_data["name"].astype(str).str.strip().str.lower()
        artist_names = songs_data["artist"].astype(str).str.strip().str.lower()

        in_songs = (
            (song_names == song_name)
            & (artist_names == artist_name)
        ).any()

        if not in_songs:
            st.warning(
                "Song and artist not found. Check the spelling "
                "or try another song."
            )
        else:
            st.subheader("Content-Based Recommendations")

            recs = content_recommendation(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=songs_data,
                transformed_data=transformed_data,
                k=k,
            )

            show_recommendations(recs)

    elif filtering_type == "Collaborative Filtering":

        # Check whether the song exists in the collaborative dataset
        collab_names = (
            filtered_data["name"].astype(str).str.strip().str.lower()
        )
        collab_artists = (
            filtered_data["artist"].astype(str).str.strip().str.lower()
        )

        in_collab_data = (
            (collab_names == song_name)
            & (collab_artists == artist_name)
        ).any()

        if not in_collab_data:
            st.warning(
                "This song was not found in the collaborative filtering "
                "dataset. Try another song."
            )
        else:
            st.subheader("Collaborative Filtering Recommendations")

            recs = collaborative_recommendation(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=filtered_data,
                track_ids=track_ids,
                interaction_matrix=interaction_matrix,
                k=k,
            )

            show_recommendations(recs)
