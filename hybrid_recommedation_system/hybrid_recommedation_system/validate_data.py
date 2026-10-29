
from pathlib import Path
import json

import pandas as pd
from scipy.sparse import load_npz


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
REPORT_DIR = BASE_DIR.parent / "reports"


def main():
    required_files = [
        "cleaned_data.csv",
        "transformed_data.npz",
        "collab_filtered_data.csv",
        "interaction_matrix.npz",
        "track_ids.npy",
        "transformed_hybrid_data.npz",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (DATA_DIR / filename).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing required data files: {missing_files}"
        )

    songs = pd.read_csv(DATA_DIR / "cleaned_data.csv")
    collaborative_songs = pd.read_csv(
        DATA_DIR / "collab_filtered_data.csv"
    )

    content_matrix = load_npz(DATA_DIR / "transformed_data.npz")
    interaction_matrix = load_npz(
        DATA_DIR / "interaction_matrix.npz"
    )
    hybrid_matrix = load_npz(
        DATA_DIR / "transformed_hybrid_data.npz"
    )

    track_ids = __import__("numpy").load(
        DATA_DIR / "track_ids.npy",
        allow_pickle=False,
    )

    checks = {
        "cleaned_data_rows_match_content_features":
            len(songs) == content_matrix.shape[0],
        "collaborative_rows_match_interaction_matrix":
            len(collaborative_songs) == interaction_matrix.shape[0],
        "collaborative_rows_match_track_ids":
            len(collaborative_songs) == len(track_ids),
        "hybrid_rows_match_collaborative_songs":
            len(collaborative_songs) == hybrid_matrix.shape[0],
    }

    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "shapes": {
            "content_features": list(content_matrix.shape),
            "interaction_matrix": list(interaction_matrix.shape),
            "hybrid_features": list(hybrid_matrix.shape),
        },
        "rows": {
            "cleaned_songs": len(songs),
            "collaborative_songs": len(collaborative_songs),
            "track_ids": len(track_ids),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "data_validation.json"
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print(json.dumps(report, indent=4))

    if not all(checks.values()):
        raise ValueError("Data validation failed. Check the report.")


if __name__ == "__main__":
    main()
