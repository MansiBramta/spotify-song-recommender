import numpy as np
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommenderSystem:
    """Mixes content-based similarity with collaborative (listening-history) similarity."""

    def __init__(self, number_of_recommendations: int, weight_content_based: float):
        if not 0 <= weight_content_based <= 1:
            raise ValueError("weight_content_based must be between 0 and 1.")
        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative = 1 - weight_content_based

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _find_song_position(songs_data, song_name, artist_name):
        """Row position of the song in songs_data (case-insensitive)."""
        song_name = song_name.lower().strip()
        artist_name = artist_name.lower().strip()
        mask = (songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)
        positions = np.flatnonzero(mask.to_numpy())
        if len(positions) == 0:
            raise ValueError(f"'{song_name}' by '{artist_name}' not found in the data.")
        return positions[0]

    @staticmethod
    def _check_alignment(songs_data, track_ids, transformed_matrix, interaction_matrix):
        """The three data structures must list the songs in exactly the same order."""
        n = len(songs_data)
        if not (len(track_ids) == transformed_matrix.shape[0] == interaction_matrix.shape[0] == n):
            raise ValueError("songs_data, track_ids and the matrices have different numbers of rows.")
        if not np.array_equal(songs_data["track_id"].to_numpy().astype(str), track_ids.astype(str)):
            raise ValueError("songs_data and track_ids are in a different order. Rebuild the data files.")

    def _calculate_content_based_similarities(self, position, transformed_matrix):
        input_vector = transformed_matrix[position]
        if not issparse(input_vector):
            input_vector = input_vector.reshape(1, -1)
        return cosine_similarity(input_vector, transformed_matrix).ravel()

    def _calculate_collaborative_filtering_similarities(self, position, interaction_matrix):
        return cosine_similarity(interaction_matrix[position], interaction_matrix).ravel()

    @staticmethod
    def _normalize_similarities(similarity_scores):
        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)
        if maximum == minimum:  # avoid dividing by zero
            return np.zeros_like(similarity_scores)
        return (similarity_scores - minimum) / (maximum - minimum)

    def _weighted_combination(self, content_based_scores, collaborative_filtering_scores):
        return (self.weight_content_based * content_based_scores
                + self.weight_collaborative * collaborative_filtering_scores)

    # --------------------------------------------------------------------- main
    def give_recommendations(self, song_name, artist_name, songs_data, track_ids,
                             transformed_matrix, interaction_matrix):
        self._check_alignment(songs_data, track_ids, transformed_matrix, interaction_matrix)
        position = self._find_song_position(songs_data, song_name, artist_name)

        content = self._calculate_content_based_similarities(position, transformed_matrix)
        collaborative = self._calculate_collaborative_filtering_similarities(position, interaction_matrix)

        scores = self._weighted_combination(
            self._normalize_similarities(content),
            self._normalize_similarities(collaborative),
        )
        scores[position] = np.inf  # the chosen song always comes first

        top = np.argsort(scores)[::-1][: self.number_of_recommendations + 1]
        return songs_data.iloc[top].drop(columns=["track_id"]).reset_index(drop=True)
