import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recommender.data import Interactions, split_by_time  # noqa: E402
from recommender.metrics import evaluate, ndcg, top_n  # noqa: E402
from recommender.models import (EASE, ItemCosine, ItemPearson, Popularity, PureSVD,  # noqa: E402
                                pearson_similarity, similar_items)


def frame(rows):
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


@pytest.fixture
def ratings():
    # Two tastes: users 1-3 rate movies 10-12, users 4-6 rate movies 20-22, everyone rates 30.
    rows, t = [], 0
    for user in range(1, 7):
        liked = [10, 11, 12] if user <= 3 else [20, 21, 22]
        for movie in liked:
            rows.append((user, movie, 4.0 + user % 2, t)); t += 1
        rows.append((user, 30, 3.0, t)); t += 1
    return frame(rows)


def test_split_holds_out_latest_ratings():
    r = frame([(1, m, 4.0, ts) for m, ts in [(1, 5), (2, 1), (3, 4), (4, 2), (5, 3)]])
    train, test = split_by_time(r, test_share=0.2)
    assert test["movieId"].tolist() == [1]
    assert len(train) == 4


def test_interactions_and_relevance(ratings):
    data = Interactions.from_frame(ratings)
    assert data.ratings.shape == (6, 7)
    held_out = frame([(1, 22, 5.0, 99), (1, 21, 2.0, 99), (2, 999, 5.0, 99)])
    relevant = data.relevant(held_out)
    assert relevant == {0: {int(np.searchsorted(data.items, 22))}}   # low rating and unknown movie dropped


def test_top_n_skips_seen_and_orders():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]])
    exclude = np.array([[0, 1, 0, 0]])
    assert top_n(scores, exclude, n=2).tolist() == [[3, 2]]


def test_ndcg():
    assert ndcg([True, False], 1) == 1.0
    assert ndcg([False, True], 1) == pytest.approx(1 / np.log2(3))
    assert ndcg([False, False], 2) == 0.0


def test_pearson_matches_pandas():
    rng = np.random.default_rng(0)
    dense = rng.integers(1, 6, size=(40, 3)).astype(float)
    dense[rng.random(dense.shape) < 0.3] = 0
    corr = pearson_similarity(dense, min_common=2)
    a, b = dense[:, 0], dense[:, 1]
    both = (a > 0) & (b > 0)
    assert corr[0, 1] == pytest.approx(np.corrcoef(a[both], b[both])[0, 1])
    assert np.allclose(corr, corr.T)


def test_pearson_ignores_pairs_with_few_common_raters():
    dense = np.array([[5, 4, 0], [1, 2, 0], [0, 0, 3]], dtype=float)
    assert np.all(pearson_similarity(dense, min_common=3) == 0)


@pytest.mark.parametrize("model", [ItemCosine(), ItemPearson(min_common=2), PureSVD(factors=2), EASE(l2=1.0)])
def test_models_recover_tastes(ratings, model):
    # Hide movie 12 from user 1; a good model should rank it above movie 22.
    train = ratings[~((ratings.userId == 1) & (ratings.movieId == 12))]
    data = Interactions.from_frame(train)
    scores = model.fit(data).score(np.array([0]))[0]
    col = {m: int(np.searchsorted(data.items, m)) for m in (12, 22)}
    assert scores[col[12]] > scores[col[22]]


def test_evaluate_popularity(ratings):
    data = Interactions.from_frame(ratings[ratings.movieId != 22])
    relevant = {0: {int(np.searchsorted(data.items, 20))}}
    result = evaluate(Popularity().fit(data), data, relevant, n=1)
    assert set(result.index) >= {"precision@1", "recall@1", "ndcg@1", "catalogue coverage"}
    assert 0 <= result["precision@1"] <= 1


def test_similar_items_excludes_itself():
    similarity = np.array([[0, 0.2, 0.9], [0.2, 0, 0.1], [0.9, 0.1, 0]])
    assert similar_items(similarity, 0, n=2).tolist() == [2, 1]
