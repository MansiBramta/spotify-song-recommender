
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def collaborative_recommendation(
    song_name,
    artist_name,
    songs_data,
    track_ids,
    interaction_matrix,
    k=5,
):
    # Ensure the track IDs match the number of matrix rows
    track_ids = np.asarray(track_ids).astype(str)

    if len(track_ids) != interaction_matrix.shape[0]:
        raise ValueError(
            "The number of track IDs does not match the rows "
            "in the interaction matrix. Rebuild the data files."
        )

    # Find the selected song using its name and artist
    names = songs_data["name"].astype(str).str.strip().str.lower()
    artists = songs_data["artist"].astype(str).str.strip().str.lower()

    mask = (
        (names == song_name.strip().lower())
        & (artists == artist_name.strip().lower())
    )

    song_row = songs_data.loc[mask]

    if song_row.empty:
        return pd.DataFrame()

    input_track_id = str(song_row.iloc[0]["track_id"])

    # Find the selected song's row in the interaction matrix
    matching_indices = np.flatnonzero(track_ids == input_track_id)

    if len(matching_indices) == 0:
        return pd.DataFrame()

    song_index = matching_indices[0]

    # Calculate similarity against all songs
    input_vector = interaction_matrix[song_index]

    similarity_scores = cosine_similarity(
        input_vector,
        interaction_matrix,
    ).ravel()

    # Rank songs by similarity, highest first
    ranked_indices = np.argsort(similarity_scores)[::-1]

    recommendations = []

    for index in ranked_indices:
        recommended_track_id = track_ids[index]

        # Do not recommend the selected song itself
        if recommended_track_id == input_track_id:
            continue

        recommendations.append({
            "track_id": recommended_track_id,
            "score": similarity_scores[index],
        })

        if len(recommendations) >= k:
            break

    if not recommendations:
        return pd.DataFrame()

    recommendation_scores = pd.DataFrame(recommendations)

    # Attach song details to the ranked recommendations
    result = songs_data.copy()
    result["track_id"] = result["track_id"].astype(str)

    recs = (
        result.drop_duplicates(subset="track_id")
        .merge(recommendation_scores, on="track_id", how="inner")
        .sort_values("score", ascending=False)
        .drop(columns="score")
        .reset_index(drop=True)
    )

    columns = [
        col for col in ["name", "artist", "spotify_preview_url"]
        if col in recs.columns
    ]

    return recs[columns].head(k)
