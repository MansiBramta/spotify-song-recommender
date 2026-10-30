from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz, issparse
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Spotify Song Recommender")
st.write(
    "Discover songs using content-based, collaborative, "
    "or hybrid recommendation methods."
)


# --------------------------------------------------
# Data directory
# --------------------------------------------------

HERE = Path(__file__).parent
DATA = next(
    (p for p in (HERE / "data", HERE.parent / "data") if p.exists()),
    HERE.parent / "data",
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading recommendation data...")
def load_data():
    return {
        "songs_data": pd.read_csv(DATA / "cleaned_data.csv"),
        "transformed_data": load_npz(DATA / "transformed_data.npz"),
        "filtered_data": pd.read_csv(DATA / "collab_filtered_data.csv"),
        "interaction_matrix": load_npz(DATA / "interaction_matrix.npz"),
    }


required_files = [
    "cleaned_data.csv",
    "transformed_data.npz",
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
        "Missing data files:\n\n"
        + "\n".join(f"- {filename}" for filename in missing_files)
        + "\n\nCheck your data folder and run build_data.py if required."
    )
    st.stop()


try:
    data = load_data()

    songs_data = data["songs_data"]
    transformed_data = data["transformed_data"]
    filtered_data = data["filtered_data"]
    interaction_matrix = data["interaction_matrix"]

except Exception as exc:
    st.error(f"Unable to load data: {exc}")
    st.stop()


# --------------------------------------------------
# Validate data
# --------------------------------------------------

required_columns = {"name", "artist", "track_id"}

for dataset_name, dataset in [
    ("Content dataset", songs_data),
    ("Collaborative dataset", filtered_data),
]:
    missing_columns = required_columns - set(dataset.columns)

    if missing_columns:
        st.error(
            f"{dataset_name} is missing columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
        st.stop()


if len(songs_data) != transformed_data.shape[0]:
    st.error(
        "The content matrix rows do not match cleaned_data.csv. "
        "Rebuild or realign your content data."
    )
    st.stop()


if len(filtered_data) != interaction_matrix.shape[0]:
    st.error(
        "The interaction matrix rows do not match "
        "collab_filtered_data.csv. Rebuild or realign your data."
    )
    st.stop()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )


def find_song_position(dataset, song_name, artist_name):

    mask = (
        normalize_text(dataset["name"]).eq(
            str(song_name).strip().casefold()
        )
        &
        normalize_text(dataset["artist"]).eq(
            str(artist_name).strip().casefold()
        )
    )

    positions = np.flatnonzero(mask.to_numpy())

    if len(positions) == 0:
        return None

    return int(positions[0])


def calculate_similarity(position, matrix):

    input_vector = matrix[position]

    if not issparse(input_vector):
        input_vector = np.asarray(input_vector).reshape(1, -1)

    scores = cosine_similarity(
        input_vector,
        matrix,
    ).ravel()

    return scores


def normalize_scores(scores):

    scores = np.asarray(scores, dtype=float)

    minimum = np.min(scores)
    maximum = np.max(scores)

    if maximum == minimum:
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def get_recommendations_from_scores(
    dataset,
    scores,
    selected_track_id,
    k,
):

    dataset = dataset.copy()
    dataset["track_id"] = dataset["track_id"].astype(str)

    scores = np.asarray(scores, dtype=float)

    dataset = dataset.reset_index(drop=True)
    dataset["score"] = scores

    # Do not recommend the selected song itself.
    dataset = dataset[
        dataset["track_id"] != str(selected_track_id)
    ]

    # Avoid duplicate tracks.
    dataset = dataset.drop_duplicates(subset="track_id")

    dataset = dataset.sort_values(
        "score",
        ascending=False,
    )

    return dataset.head(k).reset_index(drop=True)


# --------------------------------------------------
# Content-based filtering
# --------------------------------------------------

def content_recommendation_app(
    song_name,
    artist_name,
    k,
):

    position = find_song_position(
        songs_data,
        song_name,
        artist_name,
    )

    if position is None:
        return pd.DataFrame(), "Song not found in content dataset."

    scores = calculate_similarity(
        position,
        transformed_data,
    )

    selected_track_id = songs_data.iloc[position]["track_id"]

    recommendations = get_recommendations_from_scores(
        songs_data,
        scores,
        selected_track_id,
        k,
    )

    return recommendations, None


# --------------------------------------------------
# Collaborative filtering
# --------------------------------------------------

def collaborative_recommendation_app(
    song_name,
    artist_name,
    k,
):

    position = find_song_position(
        filtered_data,
        song_name,
        artist_name,
    )

    if position is None:
        return (
            pd.DataFrame(),
            "Song not found in collaborative dataset.",
        )

    scores = calculate_similarity(
        position,
        interaction_matrix,
    )

    selected_track_id = filtered_data.iloc[position]["track_id"]

    recommendations = get_recommendations_from_scores(
        filtered_data,
        scores,
        selected_track_id,
        k,
    )

    return recommendations, None


# --------------------------------------------------
# Hybrid filtering with automatic fallback
# --------------------------------------------------

def hybrid_recommendation_app(
    song_name,
    artist_name,
    content_weight,
    k,
):

    content_position = find_song_position(
        songs_data,
        song_name,
        artist_name,
    )

    collaborative_position = find_song_position(
        filtered_data,
        song_name,
        artist_name,
    )

    # CASE 1:
    # Song exists in content dataset but not collaborative dataset.
    # Use content-based filtering ONLY.

    if (
        content_position is not None
        and collaborative_position is None
    ):

        recommendations, error = content_recommendation_app(
            song_name,
            artist_name,
            k,
        )

        return (
            recommendations,
            error,
            "Content-Based Filtering Only",
        )

    # CASE 2:
    # Song exists in collaborative dataset but not content dataset.
    # Use collaborative filtering ONLY.

    if (
        content_position is None
        and collaborative_position is not None
    ):

        recommendations, error = collaborative_recommendation_app(
            song_name,
            artist_name,
            k,
        )

        return (
            recommendations,
            error,
            "Collaborative Filtering Only",
        )

    # CASE 3:
    # Song exists in neither dataset.

    if (
        content_position is None
        and collaborative_position is None
    ):

        return (
            pd.DataFrame(),
            "Song not found in either dataset.",
            "No recommendations",
        )

    # CASE 4:
    # Song exists in BOTH datasets.
    # Calculate both similarity scores.

    content_scores = calculate_similarity(
        content_position,
        transformed_data,
    )

    collaborative_scores = calculate_similarity(
        collaborative_position,
        interaction_matrix,
    )

    # Normalize scores before combining them.
    content_scores = normalize_scores(content_scores)

    collaborative_scores = normalize_scores(
        collaborative_scores
    )

    content_track_ids = (
        songs_data["track_id"].astype(str).to_numpy()
    )

    collaborative_track_ids = (
        filtered_data["track_id"].astype(str).to_numpy()
    )

    selected_track_id = str(
        songs_data.iloc[content_position]["track_id"]
    )

    # Build score tables indexed by track ID.
    content_table = pd.DataFrame({
        "track_id": content_track_ids,
        "content_score": content_scores,
    }).drop_duplicates("track_id")

    collaborative_table = pd.DataFrame({
        "track_id": collaborative_track_ids,
        "collaborative_score": collaborative_scores,
    }).drop_duplicates("track_id")

    # Combine the scores for tracks found in either dataset.
    combined = pd.merge(
        content_table,
        collaborative_table,
        on="track_id",
        how="outer",
    )

    combined["content_score"] = (
        combined["content_score"].fillna(0)
    )

    combined["collaborative_score"] = (
        combined["collaborative_score"].fillna(0)
    )

    # Apply the user-selected weights.
    collaborative_weight = 1 - content_weight

    combined["hybrid_score"] = (
        content_weight * combined["content_score"]
        +
        collaborative_weight * combined["collaborative_score"]
    )

    # Remove the selected song.
    combined = combined[
        combined["track_id"] != selected_track_id
    ]

    # Attach song details, preferring content metadata.
    content_metadata = (
        songs_data.copy()
        .assign(track_id=songs_data["track_id"].astype(str))
        .drop_duplicates("track_id")
    )

    collaborative_metadata = (
        filtered_data.copy()
        .assign(track_id=filtered_data["track_id"].astype(str))
        .drop_duplicates("track_id")
    )

    metadata = content_metadata.merge(
        collaborative_metadata,
        on="track_id",
        how="outer",
        suffixes=("", "_collab"),
    )

    # Fill missing content metadata from the collaborative dataset.
    for column in ["name", "artist", "spotify_preview_url"]:

        collab_column = f"{column}_collab"

        if collab_column in metadata.columns:

            if column not in metadata.columns:
                metadata[column] = metadata[collab_column]

            else:
                metadata[column] = metadata[column].fillna(
                    metadata[collab_column]
                )

    recommendations = combined.merge(
        metadata,
        on="track_id",
        how="left",
    )

    recommendations = recommendations.sort_values(
        "hybrid_score",
        ascending=False,
    ).head(k)

    source = (
        f"Hybrid Filtering "
        f"({content_weight:.0%} Content + "
        f"{collaborative_weight:.0%} Collaborative)"
    )

    return recommendations.reset_index(drop=True), None, source


# --------------------------------------------------
# Display recommendations
# --------------------------------------------------

def show_recommendations(recommendations):

    if recommendations.empty:
        st.warning("No recommendations found.")
        return

    for index, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1,
    ):

        song = str(row.get("name", "Unknown Song"))
        artist = str(row.get("artist", "Unknown Artist"))

        st.markdown(f"### {index}. {song}")
        st.write(f"**Artist:** {artist}")

        if "hybrid_score" in row and pd.notna(row["hybrid_score"]):
            st.caption(
                f"Hybrid score: {row['hybrid_score']:.4f}"
            )

        if "content_score" in row and pd.notna(row["content_score"]):
            st.caption(
                f"Content score: {row['content_score']:.4f}"
            )

        if (
            "collaborative_score" in row
            and pd.notna(row["collaborative_score"])
        ):
            st.caption(
                f"Collaborative score: "
                f"{row['collaborative_score']:.4f}"
            )

        url = row.get("spotify_preview_url")

        if isinstance(url, str) and url.strip():
            st.audio(url)
        else:
            st.caption("No preview available")

        st.divider()


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.sidebar.header("Recommendation Settings")

filtering_type = st.sidebar.radio(
    "Choose a recommendation method:",
    [
        "Content-Based Filtering",
        "Collaborative Filtering",
        "Hybrid Filtering",
    ],
)

k = st.sidebar.selectbox(
    "Number of recommendations:",
    [5, 10, 15, 20],
    index=1,
)

content_weight = 0.5

if filtering_type == "Hybrid Filtering":

    content_weight = (
        st.sidebar.slider(
            "Content-Based Weight (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
        )
        / 100
    )

    collaborative_weight = 1 - content_weight

    st.sidebar.write(
        f"Content-Based: {content_weight:.0%}"
    )

    st.sidebar.write(
        f"Collaborative: {collaborative_weight:.0%}"
    )


# --------------------------------------------------
# Song selection
# --------------------------------------------------

# Include songs from both datasets so content-only songs
# remain selectable in hybrid mode.

@st.cache_resource(show_spinner=False)
def build_song_catalogue():
    catalogue = pd.concat(
        [songs_data[["name", "artist"]], filtered_data[["name", "artist"]]],
        ignore_index=True,
    ).fillna("").drop_duplicates()
    catalogue["label"] = catalogue["name"].astype(str) + " — " + catalogue["artist"].astype(str)
    return catalogue.sort_values("label").reset_index(drop=True)


song_catalogue = build_song_catalogue()

query = st.text_input("Search for a song or artist:", placeholder="e.g. whenever, wherever")

if not query.strip():
    st.info("Type part of a song or artist name to begin.")
    st.stop()

matches = song_catalogue[
    song_catalogue["label"].str.contains(query.strip(), case=False, regex=False)
].head(50)

if matches.empty:
    st.warning("No records for this song. Try something else?")
    st.stop()

selected_label = st.selectbox("Select a song and artist:", matches["label"].tolist())
selected_song_row = matches[matches["label"] == selected_label].iloc[0]

song_name = str(selected_song_row["name"])
artist_name = str(selected_song_row["artist"])


# Show availability in both datasets.
in_content = (
    find_song_position(
        songs_data,
        song_name,
        artist_name,
    )
    is not None
)

in_collaborative = (
    find_song_position(
        filtered_data,
        song_name,
        artist_name,
    )
    is not None
)

col1, col2 = st.columns(2)

col1.metric(
    "Available in content data",
    "Yes" if in_content else "No",
)

col2.metric(
    "Available in collaborative data",
    "Yes" if in_collaborative else "No",
)


# --------------------------------------------------
# Generate recommendations
# --------------------------------------------------

if st.button(
    "Get Recommendations",
    type="primary",
    use_container_width=True,
):

    try:

        with st.spinner("Generating recommendations..."):

            if filtering_type == "Content-Based Filtering":

                recommendations, error = (
                    content_recommendation_app(
                        song_name,
                        artist_name,
                        k,
                    )
                )

                source = "Content-Based Filtering"

            elif filtering_type == "Collaborative Filtering":

                recommendations, error = (
                    collaborative_recommendation_app(
                        song_name,
                        artist_name,
                        k,
                    )
                )

                source = "Collaborative Filtering"

            else:

                recommendations, error, source = (
                    hybrid_recommendation_app(
                        song_name,
                        artist_name,
                        content_weight,
                        k,
                    )
                )

        if error:

            st.warning(error)

        else:

            st.success(
                f"Recommendation source: {source}"
            )

            show_recommendations(recommendations)

            

    except Exception as exc:

        st.error(
            f"An error occurred while generating recommendations: {exc}"
        )