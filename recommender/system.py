"""
pairwise-recommender-system - RecommenderSystem

A minimal recommendation engine providing:

* sparse, normalized cosine similarity;
* user-based and item-based collaborative filtering with neighbor
  selection and proper rating prediction;
* weighted content-based recommendation driven by a user profile;
* cold-start handling via popularity, diversity and exploration;
* ranking by relevance with diversity re-ranking and business rules
  (ad insertion).

Rating / feature vectors are sparse ``dict`` mappings (id -> value).
Coordinates that are absent mean "unknown" and are never treated as 0.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Optional


class RecommenderSystem:
    def __init__(self, ratings: Optional[dict] = None,
                 item_features: Optional[dict] = None):
        # ratings: {user_id: {item_id: rating}}
        self.ratings = ratings if ratings is not None else {}
        # item_features: {item_id: {feature: value}}
        self.item_features = (
            item_features if item_features is not None else {}
        )

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------
    def calculate_similarity(self, vec_a: Optional[dict] = None,
                             vec_b: Optional[dict] = None) -> float:
        """Cosine similarity between two sparse vectors.

        * Cosine instead of Euclidean distance, so magnitude does not
          dominate.
        * Each vector is L2-normalized, so a user who rates everything
          highly is not automatically "similar" to everyone.
        * Sparse handling: only coordinates present in a vector enter its
          norm and only shared coordinates enter the dot product. A
          missing rating is unknown, not 0.
        """
        if not vec_a or not vec_b:
            return 0.0

        common = vec_a.keys() & vec_b.keys()
        if not common:
            return 0.0

        dot = sum(vec_a[key] * vec_b[key] for key in common)
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Collaborative filtering
    # ------------------------------------------------------------------
    def _user_vector(self, ratings: dict, user) -> dict:
        return ratings.get(user, {})

    def _item_vector(self, ratings: dict, item) -> dict:
        return {user: user_ratings[item]
                for user, user_ratings in ratings.items()
                if item in user_ratings}

    @staticmethod
    def _neighbors(similarities: dict, k: int):
        """Select the k nearest, positively correlated neighbors."""
        positive = [(other, sim) for other, sim in similarities.items()
                    if sim > 0.0]
        positive.sort(key=lambda pair: pair[1], reverse=True)
        return positive[:k] if k and k > 0 else positive

    def _predict_user_based(self, ratings: dict, user, item,
                            k: int) -> Optional[float]:
        if user not in ratings:
            return None
        user_mean = self._mean(self._user_vector(ratings, user))
        target_vec = self._user_vector(ratings, user)

        similarities = {}
        for other, other_ratings in ratings.items():
            if other == user or item not in other_ratings:
                continue
            sim = self.calculate_similarity(target_vec, other_ratings)
            if sim > 0.0:
                similarities[other] = sim

        neighbors = self._neighbors(similarities, k)
        if not neighbors:
            return user_mean

        weighted_sum = 0.0
        weight_total = 0.0
        for neighbor, sim in neighbors:
            neighbor_mean = self._mean(ratings[neighbor])
            weighted_sum += sim * (ratings[neighbor][item] - neighbor_mean)
            weight_total += abs(sim)
        if weight_total == 0.0:
            return user_mean
        return user_mean + weighted_sum / weight_total

    def _predict_item_based(self, ratings: dict, user, item,
                            k: int) -> Optional[float]:
        if user not in ratings or not ratings[user]:
            return None
        target_vec = self._item_vector(ratings, item)
        if not target_vec:
            return self._mean(ratings[user])

        similarities = {}
        for rated_item in ratings[user]:
            if rated_item == item:
                continue
            sim = self.calculate_similarity(
                target_vec, self._item_vector(ratings, rated_item))
            if sim > 0.0:
                similarities[rated_item] = sim

        neighbors = self._neighbors(similarities, k)
        if not neighbors:
            return self._mean(ratings[user])

        weighted_sum = sum(sim * ratings[user][rated_item]
                           for rated_item, sim in neighbors)
        weight_total = sum(abs(sim) for _, sim in neighbors)
        if weight_total == 0.0:
            return self._mean(ratings[user])
        return weighted_sum / weight_total

    @staticmethod
    def _mean(vector: dict) -> float:
        return sum(vector.values()) / len(vector) if vector else 0.0

    def collaborative_filtering(self, ratings: Optional[dict] = None,
                                user=None, item=None, method: str = "user",
                                k: int = 5, top_n: Optional[int] = None):
        """User-based or item-based collaborative filtering.

        ``method`` selects ``"user"`` (user-based) or ``"item"``
        (item-based) neighborhoods. Only the ``k`` most similar
        neighbors contribute, and predictions are weighted rating
        estimates (mean-centered for user-based CF) rather than raw
        similarities.

        * ``user`` + ``item`` -> predicted rating (float).
        * ``user`` only       -> ranked list of ``(item, score)`` for
          items the user has not rated.
        """
        if ratings is None:
            ratings = self.ratings
        if not ratings:
            return []
        if method not in ("user", "item"):
            raise ValueError("method must be 'user' or 'item'")

        predictor = (self._predict_user_based if method == "user"
                     else self._predict_item_based)

        if item is not None:
            if user is None:
                raise ValueError("a user is required to predict a rating")
            return predictor(ratings, user, item, k)

        if user is None:
            return []

        rated = set(ratings.get(user, {}))
        all_items = {candidate
                     for user_ratings in ratings.values()
                     for candidate in user_ratings}
        predictions = []
        for candidate in all_items - rated:
            score = predictor(ratings, user, candidate, k)
            if score is not None:
                predictions.append((candidate, score))
        predictions.sort(key=lambda pair: pair[1], reverse=True)
        return predictions[:top_n] if top_n else predictions

    # ------------------------------------------------------------------
    # Content-based recommendation
    # ------------------------------------------------------------------
    def build_user_profile(self, ratings: dict, user,
                           item_features: dict) -> dict:
        """Build a weighted user profile from the user's rated items.

        Item features are weighted by the user's (mean-centered) rating,
        so liked content dominates the profile.
        """
        user_ratings = ratings.get(user, {})
        if not user_ratings:
            return {}
        mean = self._mean(user_ratings)
        profile = defaultdict(float)
        weight_total = 0.0
        for item_id, rating in user_ratings.items():
            features = item_features.get(item_id, {})
            weight = rating - mean
            for feature, value in features.items():
                profile[feature] += weight * value
            weight_total += abs(weight)
        if weight_total:
            for feature in profile:
                profile[feature] /= weight_total
        return dict(profile)

    def content_based_recommend(self, user_profile: Optional[dict] = None,
                                items: Optional[dict] = None,
                                weights: Optional[dict] = None,
                                ratings: Optional[dict] = None,
                                user=None, top_n: Optional[int] = None,
                                exclude_rated: bool = True):
        """Recommend items by weighted feature similarity to a profile.

        Every item has its own feature vector; every feature may carry
        its own weight. The user profile can be supplied directly or
        learned from the user's rating history via
        :meth:`build_user_profile`.
        """
        if items is None:
            items = self.item_features
        if not items:
            return []

        if user_profile is None and user is not None:
            user_profile = self.build_user_profile(
                ratings if ratings is not None else self.ratings, user, items)
        if not user_profile:
            return []

        rated = set()
        if exclude_rated and user is not None:
            rated = set((ratings if ratings is not None
                         else self.ratings).get(user, {}))

        scores = []
        for item_id, features in items.items():
            if item_id in rated:
                continue
            if not features:
                continue
            dot = norm_p = norm_i = 0.0
            for feature, profile_value in user_profile.items():
                item_value = features.get(feature)
                if item_value is None:
                    continue  # sparse: missing feature is unknown, not 0
                feature_weight = (weights.get(feature, 1.0)
                                  if weights else 1.0)
                dot += feature_weight * profile_value * item_value
                norm_p += feature_weight * profile_value * profile_value
            for value in features.values():
                norm_i += value * value
            if norm_p > 0.0 and norm_i > 0.0:
                scores.append((item_id, dot / math.sqrt(norm_p * norm_i)))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_n] if top_n else scores

    # ------------------------------------------------------------------
    # Cold start
    # ------------------------------------------------------------------
    def _popular_items(self, ratings: dict, item_features: dict):
        counts = defaultdict(int)
        sums = defaultdict(float)
        for user_ratings in ratings.values():
            for item_id, rating in user_ratings.items():
                counts[item_id] += 1
                sums[item_id] += rating
        scored = [
            (item_id, sums[item_id] / counts[item_id], counts[item_id])
            for item_id in counts
        ]
        scored.sort(key=lambda triple: (triple[2], triple[1]), reverse=True)
        return scored

    def handle_cold_start(self, user=None, ratings: Optional[dict] = None,
                          item_features: Optional[dict] = None,
                          top_n: int = 10, exploration_ratio: float = 0.3,
                          seed: Optional[int] = 42):
        """Cold-start strategy for new users and new items.

        * New user (no ratings): popularity-based recommendations so the
          home page is never empty.
        * New item (no ratings yet): guaranteed exposure through the
          exploration slot, so fresh content is not buried.
        * Exploration: a fraction of slots is reserved for diverse /
          unknown items instead of always repeating known hits.
        """
        if ratings is None:
            ratings = self.ratings
        if item_features is None:
            item_features = self.item_features

        catalog = list(item_features.keys())
        is_new_user = user is None or not ratings.get(user)
        popular = [item_id for item_id, _, _
                   in self._popular_items(ratings, item_features)]

        rated = set(ratings.get(user, {})) if user is not None else set()
        unseen_new_items = [item_id for item_id in catalog
                            if item_id not in rated
                            and item_id not in ratings_known(ratings)]
        rng = random.Random(seed)
        rng.shuffle(unseen_new_items)

        if not is_new_user:
            known_unseen = [item_id for item_id in popular
                            if item_id not in rated]
            pool = known_unseen + unseen_new_items
            return pool[:top_n]

        # New user: mostly popular, partly exploration.
        n_explore = min(len(unseen_new_items),
                        int(round(top_n * exploration_ratio)))
        results = popular[:top_n - n_explore]
        results.extend(unseen_new_items[:n_explore])

        # Backfill from the catalog (diverse fillers) if data is sparse.
        if len(results) < top_n:
            fillers = [item_id for item_id in catalog
                       if item_id not in results]
            rng.shuffle(fillers)
            results.extend(fillers[:top_n - len(results)])
        return results[:top_n]

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    def rank_recommendations(self, candidates: Optional[list] = None,
                             diversity_weight: float = 0.3,
                             ads: Optional[list] = None,
                             ad_every: int = 5,
                             top_n: Optional[int] = None):
        """Rank candidates by relevance, diversity and business rules.

        ``candidates`` is a list of dicts such as::

            {"item_id": 1, "score": 0.9, "category": "music"}

        * Relevance: ordered by the prediction/score instead of time.
        * Diversity: greedy re-ranking penalizes repeatedly picking the
          same category.
        * Business rules: sponsored items from ``ads`` are inserted every
          ``ad_every`` organic positions.
        """
        if not candidates:
            return []

        def relevance(candidate):
            return candidate.get("score", candidate.get("relevance", 0.0))

        ordered = sorted(candidates, key=relevance, reverse=True)

        # Greedy diversity re-rank: relevance minus a penalty for
        # categories already shown.
        remaining = ordered[:]
        ranked = []
        seen_categories = set()
        while remaining:
            def objective(candidate):
                category = candidate.get("category")
                penalty = diversity_weight if category in seen_categories \
                    else 0.0
                return relevance(candidate) - penalty

            best = max(remaining, key=objective)
            remaining.remove(best)
            ranked.append(best)
            seen_categories.add(best.get("category"))

        if top_n is not None:
            ranked = ranked[:top_n]

        results = []
        if ads:
            for position, candidate in enumerate(ranked, start=1):
                results.append(candidate.get("item_id", candidate))
                if position % ad_every == 0:
                    ad_index = (position // ad_every - 1) % len(ads)
                    results.append(ads[ad_index])
        else:
            results = [candidate.get("item_id", candidate)
                       for candidate in ranked]
        return results


def ratings_known(ratings: dict) -> set:
    """Set of item ids that appear in at least one user's history."""
    return {item_id for user_ratings in ratings.values()
            for item_id in user_ratings}
