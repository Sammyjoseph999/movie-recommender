"""Ranking metrics for top-N recommendation."""
import numpy as np
import pandas as pd


def top_n(scores, exclude, n=10):
    """The `n` highest-scoring columns per row, best first, skipping excluded entries."""
    scores = np.array(scores, dtype=float)
    scores[exclude > 0] = -np.inf
    top = np.argpartition(-scores, n, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def ndcg(hits, n_relevant):
    """Normalised discounted cumulative gain of one ranked list of hits (True/False)."""
    discounts = 1 / np.log2(np.arange(2, len(hits) + 2))
    ideal = discounts[:min(n_relevant, len(hits))].sum()
    return float((np.asarray(hits) * discounts).sum() / ideal) if ideal else 0.0


def evaluate(model, data, relevant, n=10, batch=128):
    """Precision, recall and NDCG at `n`, plus how much of the catalogue gets recommended.

    `relevant` maps a user row to the set of held-out movie columns that user liked.
    Recall is divided by min(n, relevant) so that a perfect list scores 1.
    """
    seen = data.seen
    popularity = np.asarray(seen.sum(axis=0)).ravel()
    users = np.array(sorted(relevant))
    precision, recall, gain, mean_popularity = [], [], [], []
    recommended = set()
    for start in range(0, len(users), batch):
        rows = users[start:start + batch]
        lists = top_n(model.score(rows), seen[rows].toarray(), n)
        for row, items in zip(rows, lists):
            hits = np.array([item in relevant[row] for item in items])
            precision.append(hits.mean())
            recall.append(hits.sum() / min(n, len(relevant[row])))
            gain.append(ndcg(hits, len(relevant[row])))
            mean_popularity.append(popularity[items].mean())
            recommended.update(items.tolist())
    return pd.Series({
        f"precision@{n}": np.mean(precision),
        f"recall@{n}": np.mean(recall),
        f"ndcg@{n}": np.mean(gain),
        "catalogue coverage": len(recommended) / seen.shape[1],
        "ratings per recommended movie": np.mean(mean_popularity),
    })
