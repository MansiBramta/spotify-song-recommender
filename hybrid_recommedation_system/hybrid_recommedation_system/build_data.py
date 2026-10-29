"""
Run this ONCE (from the project root) to create every file the app needs:

    python build_data.py

It downloads the dataset with kagglehub, cleans it, builds the feature matrices
and the user-song interaction matrix, and saves everything into ./data
"""
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from category_encoders.count import CountEncoder
from scipy.sparse import coo_matrix, csr_matrix, save_npz
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# a song needs at least this many listeners to be used by the collaborative part
MIN_LISTENERS = 50

FEATURE_COLS = [
    "year", "artist", "time_signature", "key", "tags",
    "duration_ms", "loudness", "tempo",
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "mode",
]


def build_features(df):
    """Turn the songs table into a sparse numeric feature matrix."""
    transformer = ColumnTransformer(
        transformers=[
            ("frequency_encode", CountEncoder(normalize=True, return_df=True), ["year"]),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ["artist", "time_signature", "key"]),
            ("tfidf", TfidfVectorizer(max_features=85), "tags"),
            ("standard_scale", StandardScaler(), ["duration_ms", "loudness", "tempo"]),
            ("min_max_scale", MinMaxScaler(),
             ["danceability", "energy", "speechiness", "acousticness",
              "instrumentalness", "liveness", "valence"]),
        ],
        remainder="passthrough",  # 'mode'
    )
    return csr_matrix(transformer.fit_transform(df[FEATURE_COLS]), dtype=np.float32)


def main():
    path = Path(kagglehub.dataset_download("undefinenull/million-song-dataset-spotify-lastfm"))

    # ------------------------------------------------------------------ songs
    print("Cleaning songs data...")
    songs = pd.read_csv(path / "Music Info.csv")
    songs = songs.dropna(subset=["name", "artist"])
    songs["name"] = songs["name"].str.lower().str.strip()
    songs["artist"] = songs["artist"].str.lower().str.strip()
    songs["tags"] = songs["tags"].fillna("no_tags")
    songs = (
        songs.drop_duplicates(subset="track_id")
        .drop_duplicates(subset=["name", "artist"])
        .drop(columns=["spotify_id", "genre"])
        .reset_index(drop=True)
    )
    songs.to_csv(DATA_DIR / "cleaned_data.csv", index=False)

    print("Building content features...")
    save_npz(DATA_DIR / "transformed_data.npz", build_features(songs))

    # ----------------------------------------------------------- interactions
    print("Reading listening history (this takes a few minutes)...")
    inter = pd.read_csv(
        path / "User Listening History.csv",
        usecols=["track_id", "user_id", "playcount"],
        dtype={"track_id": "category", "user_id": "category", "playcount": "int32"},
    )

    # row number of each interaction's song inside `songs` (-1 = song not in songs)
    track_codes = inter["track_id"].cat.set_categories(songs["track_id"]).cat.codes.to_numpy()
    user_codes = inter["user_id"].cat.codes.to_numpy()
    n_users = len(inter["user_id"].cat.categories)
    playcounts = inter["playcount"].to_numpy()
    valid = track_codes >= 0

    print("Building interaction matrix...")
    matrix = coo_matrix(
        (playcounts[valid].astype(np.float32), (track_codes[valid], user_codes[valid])),
        shape=(len(songs), n_users),
    ).tocsr()                          # duplicate (song, user) pairs are summed
    matrix.data = np.log1p(matrix.data)  # stop heavy listeners from dominating

    # keep songs with enough listeners
    keep = matrix.getnnz(axis=1) >= MIN_LISTENERS
    filtered = songs[keep].reset_index(drop=True)
    interaction = matrix[keep]

    filtered.to_csv(DATA_DIR / "collab_filtered_data.csv", index=False)
    save_npz(DATA_DIR / "interaction_matrix.npz", interaction)
    np.save(DATA_DIR / "track_ids.npy", filtered["track_id"].to_numpy().astype(str))

    print("Building hybrid features...")
    save_npz(DATA_DIR / "transformed_hybrid_data.npz", build_features(filtered))

    print(f"Done. {len(songs):,} songs, {len(filtered):,} with enough listening history.")


if __name__ == "__main__":
    main()
