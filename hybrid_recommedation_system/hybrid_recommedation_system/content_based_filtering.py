import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def content_recommendation(song_name, artist_name, songs_data, transformed_data, k=10):
    """Return the chosen song (row 0) followed by the k most similar songs."""
    song_name = song_name.lower().strip()
    artist_name = artist_name.lower().strip()

    mask = (songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)
    positions = np.flatnonzero(mask.to_numpy())
    if len(positions) == 0:
        raise ValueError("Song not found in the dataset.")
    pos = positions[0]

    scores = cosine_similarity(transformed_data[pos], transformed_data).ravel()
    scores[pos] = np.inf  # make sure the chosen song comes first
    top = np.argsort(scores)[::-1][: k + 1]

    return songs_data.iloc[top][["name", "artist", "spotify_preview_url"]].reset_index(drop=True)
