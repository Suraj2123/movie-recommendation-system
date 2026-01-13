import pandas as pd

from mrs.data.preprocess import preprocess


def test_preprocess_types_and_missing():
    ratings = pd.DataFrame(
        {"userId": [1, 2], "movieId": [10, 20], "rating": [4.0, 5.0], "timestamp": [1, 2]}
    )
    movies = pd.DataFrame({"movieId": [10, 20], "title": ["A", "B"], "genres": ["X|Y", None]})

    out = preprocess(ratings, movies)

    assert out.ratings["userId"].dtype.kind in {"i", "u"}
    assert out.ratings["movieId"].dtype.kind in {"i", "u"}
    assert out.movies["genres"].isna().sum() == 0
