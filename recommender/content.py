"""Recommending from movie metadata (genres and tags), alone and alongside a ratings model."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def tags_before_split(tags, train):
    """Keep only tags written before the tagging user's own split point.

    A user tags a movie they have watched, so a tag written later would leak
    that user's held-out movies into the features.
    """
    split_time = train.groupby("userId")["timestamp"].max().rename("split_time")
    tags = tags.join(split_time, on="userId")
    return tags[tags["timestamp"] <= tags["split_time"]].drop(columns="split_time")


def item_features(movies, tags, items, tag_weight=0.7):
    """One unit-length row per movie in `items`: genre indicators plus TF-IDF weighted tags."""
    genres = (movies.set_index("movieId")["genres"].reindex(items).fillna("")
              .str.replace("(no genres listed)", "", regex=False).str.replace("|", " ", regex=False))
    tag_text = (tags.assign(tag=tags["tag"].str.lower().str.replace(r"\s+", "_", regex=True))
                .groupby("movieId")["tag"].apply(" ".join).reindex(items).fillna(""))
    genre_matrix = TfidfVectorizer(token_pattern=r"[^ ]+", use_idf=False).fit_transform(genres)
    tag_matrix = TfidfVectorizer(token_pattern=r"[^ ]+", min_df=2, sublinear_tf=True).fit_transform(tag_text)
    features = sparse.hstack([genre_matrix, tag_weight * tag_matrix]).tocsr()
    return normalize(features), pd.Series({"movies with tags": float((tag_text != "").mean()),
                                           "genre labels": genre_matrix.shape[1],
                                           "distinct tags used": tag_matrix.shape[1]})


class ContentModel:
    """Score movies by cosine similarity to the average features of the movies a user rated 4 or more."""
    name = "Genres and tags"

    def __init__(self, features, liked=4.0):
        self.features = features
        self.liked = liked

    def fit(self, data):
        liked = (data.ratings >= self.liked).astype(float)
        self.profiles = normalize(liked @ self.features)
        return self

    def score(self, rows):
        return np.asarray((self.profiles[rows] @ self.features.T).todense())


def standardise(scores):
    scores = np.asarray(scores, dtype=float)
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-9)


class Blend:
    """A weighted sum of two models' scores, each standardised per user first."""

    def __init__(self, main, extra, weight):
        self.main, self.extra, self.weight = main, extra, weight
        self.name = f"{main.name} + {extra.name} (weight {weight})"

    def fit(self, data):
        self.main.fit(data)
        self.extra.fit(data)
        return self

    def score(self, rows):
        return (1 - self.weight) * standardise(self.main.score(rows)) + self.weight * standardise(self.extra.score(rows))


class NewItemSlots:
    """The main model's top n - k movies, followed by the k best content matches among unrated movies."""

    def __init__(self, main, content, k=1, n=10):
        self.main, self.content, self.k, self.n = main, content, k, n
        self.name = f"{main.name} + {k} slot{'s' if k > 1 else ''} for unrated movies"

    def fit(self, data):
        self.main.fit(data)
        self.content.fit(data)
        self.seen = data.seen
        self.unrated = np.asarray(data.seen.sum(axis=0)).ravel() == 0
        return self

    def score(self, rows):
        main = np.array(self.main.score(rows), dtype=float)
        content = np.array(self.content.score(rows), dtype=float)
        main[(self.seen[rows].toarray() > 0) | self.unrated[None, :]] = -np.inf
        content[:, ~self.unrated] = -np.inf
        # Only the order matters to the metrics, so build scores that produce the intended list
        out = np.full(main.shape, -1e9)
        for r in range(len(rows)):
            top = np.argsort(-main[r])[:self.n - self.k]
            out[r, top] = 2e6 - np.arange(len(top))
            new = np.argsort(-content[r])[:self.k]
            out[r, new] = 1e6 - np.arange(len(new))
        return out
