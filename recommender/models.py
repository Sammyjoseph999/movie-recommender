"""Recommenders that score every movie for a batch of users.

Each model is fitted on an `Interactions` object and returns a dense score matrix
(users x movies) from `score(rows)`. Higher means more likely to be recommended.
"""
import numpy as np
from scipy.sparse.linalg import svds


class Popularity:
    """Recommend the most-rated movies to everyone."""
    name = "Most popular"

    def fit(self, data):
        self.counts = np.asarray(data.seen.sum(axis=0)).ravel()
        return self

    def score(self, rows):
        return np.tile(self.counts, (len(rows), 1))


class ItemCosine:
    """Movies watched by the same people as the user's movies, with cosine similarity.

    Only whether a user rated a movie is used, not the rating itself.
    """
    name = "Item-item cosine"

    def fit(self, data):
        self.seen = data.seen.toarray()
        co = self.seen.T @ self.seen
        norms = np.sqrt(np.diag(co)) + 1e-9
        self.similarity = co / np.outer(norms, norms)
        np.fill_diagonal(self.similarity, 0)
        return self

    def score(self, rows):
        return self.seen[rows] @ self.similarity


def pearson_similarity(ratings, min_common=10):
    """Item-item Pearson correlation of ratings, computed over the users who rated both.

    Pairs with fewer than `min_common` shared raters are set to zero: a correlation from
    a handful of people is mostly noise, and is often a perfect 1 or -1.
    """
    n_items = ratings.shape[1]
    # A movie with fewer than `min_common` raters cannot reach the threshold with anything,
    # so the correlations are only computed among the rest. This keeps the matrices small.
    active = np.flatnonzero((ratings > 0).sum(axis=0) >= min_common)
    ratings = ratings[:, active]
    rated = (ratings > 0).astype(float)
    common = rated.T @ rated
    mean = (ratings.T @ rated) / np.maximum(common, 1)        # mean of item i over users who rated i and j
    sum_xy = ratings.T @ ratings
    sum_x2 = (ratings ** 2).T @ rated
    covariance = sum_xy - mean * mean.T * common
    var_i = sum_x2 - mean ** 2 * common
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = covariance / np.sqrt(var_i * var_i.T)
    corr = np.nan_to_num(np.clip(corr, -1, 1))
    corr[common < min_common] = 0
    np.fill_diagonal(corr, 0)
    full = np.zeros((n_items, n_items))
    full[np.ix_(active, active)] = corr
    return full


class ItemPearson:
    """The classic approach: movies whose ratings correlate with those of movies the user liked."""
    name = "Item-item Pearson"

    def __init__(self, min_common=10, liked=4.0):
        self.min_common = min_common
        self.liked = liked

    def fit(self, data):
        dense = data.ratings.toarray()
        self.liked_matrix = (dense >= self.liked).astype(float)
        self.similarity = pearson_similarity(dense, self.min_common)
        return self

    def score(self, rows):
        return self.liked_matrix[rows] @ self.similarity


class PureSVD:
    """A truncated SVD of the ratings matrix, with unrated entries as zero (Cremonesi et al., 2010)."""
    name = "PureSVD"

    def __init__(self, factors=20):
        self.factors = factors

    def fit(self, data):
        u, s, vt = svds(data.ratings.astype(float), k=self.factors, random_state=0)
        self.user_factors = u * s
        self.item_factors = vt
        return self

    def score(self, rows):
        return self.user_factors[rows] @ self.item_factors


class EASE:
    """A linear item-to-item model with a closed-form solution (Steck, 2019).

    It learns weights B with zero diagonal that best reconstruct the user-by-movie matrix X
    as X @ B, with an L2 penalty `l2` on B.
    """
    name = "EASE"

    def __init__(self, l2=300.0):
        self.l2 = l2

    def fit(self, data):
        self.seen = data.seen.toarray()
        gram = self.seen.T @ self.seen + self.l2 * np.eye(self.seen.shape[1])
        inverse = np.linalg.inv(gram)
        self.weights = -inverse / np.diag(inverse)
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, rows):
        return self.seen[rows] @ self.weights


def similar_items(similarity, item, n=10):
    """Column indices of the `n` items most similar to `item`."""
    scores = similarity[item].astype(float).copy()
    scores[item] = -np.inf
    return np.argsort(-scores)[:n]
