"""Downloading MovieLens and splitting each user's ratings by time."""
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download(data_dir="data"):
    """Fetch ml-latest-small into `data_dir` unless it is already there."""
    folder = Path(data_dir) / "ml-latest-small"
    if not (folder / "ratings.csv").exists():
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        zipfile.ZipFile(io.BytesIO(response.content)).extractall(data_dir)
    return folder


def load(data_dir="data"):
    folder = download(data_dir)
    return pd.read_csv(folder / "ratings.csv"), pd.read_csv(folder / "movies.csv")


def split_by_time(ratings, test_share=0.2):
    """Hold out the latest `test_share` of each user's ratings.

    The model sees what a user rated first and is asked to predict what they rated next,
    which is closer to how a recommender is used than a random split.
    """
    ratings = ratings.sort_values(["userId", "timestamp", "movieId"])
    position = ratings.groupby("userId").cumcount()
    count = ratings.groupby("userId")["userId"].transform("size")
    is_test = position >= np.ceil(count * (1 - test_share))
    return ratings[~is_test], ratings[is_test]


@dataclass
class Interactions:
    """A user-by-movie ratings matrix with the mappings back to MovieLens ids."""
    ratings: sparse.csr_matrix
    users: np.ndarray
    items: np.ndarray

    @classmethod
    def from_frame(cls, frame, users=None, items=None):
        """Build the matrix from a ratings frame.

        By default the rows and columns are the users and movies that appear in `frame`.
        Passing `items` (for example every movie in movies.csv) keeps columns for movies
        with no ratings yet, so that a model using other information can recommend them.
        """
        users = np.sort(frame["userId"].unique()) if users is None else np.sort(np.asarray(users))
        items = np.sort(frame["movieId"].unique()) if items is None else np.sort(np.asarray(items))
        rows = np.searchsorted(users, frame["userId"].to_numpy())
        cols = np.searchsorted(items, frame["movieId"].to_numpy())
        matrix = sparse.csr_matrix((frame["rating"].to_numpy(float), (rows, cols)),
                                   shape=(len(users), len(items)))
        return cls(matrix, users, items)

    @property
    def seen(self):
        """1 where a user has rated a movie, whatever the rating."""
        return (self.ratings > 0).astype(float).tocsr()

    def relevant(self, held_out, min_rating=4.0):
        """For each user row, the column indices of held-out movies they rated `min_rating` or more.

        Movies that do not appear in the training data cannot be recommended and are dropped.
        """
        liked = held_out[(held_out["rating"] >= min_rating) & held_out["movieId"].isin(self.items)
                         & held_out["userId"].isin(self.users)]
        rows = np.searchsorted(self.users, liked["userId"].to_numpy())
        cols = np.searchsorted(self.items, liked["movieId"].to_numpy())
        relevant = {}
        for row, col in zip(rows, cols):
            relevant.setdefault(int(row), set()).add(int(col))
        return relevant
