"""
pairwise-recommender-system - RecommenderSystem

A small recommender system with:
- cosine similarity with mean normalization and sparse handling
- user-based / item-based collaborative filtering with neighbor
  selection and rating prediction
- content-based recommendations with weighted features and user profiles
- cold-start strategy with popularity fallback and exploration
- relevance ranking with diversity and business rules (ad insertion)
"""

import math
import random


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class RecommenderSystem:
    def __init__(self, ratings=None, item_features=None, feature_weights=None,
                 item_categories=None, item_timestamps=None):
        self.state = {}
        # Sparse ratings: {user_id: {item_id: rating}}. Missing entries are
        # "unknown", NOT zero.
        self.ratings = ratings or {}
        # Content features: {item_id: {feature: value}}
        self.item_features = item_features or {}
        # Feature importance: {feature: weight}
        self.feature_weights = feature_weights or {}
        self.item_categories = item_categories or {}
        self.item_timestamps = item_timestamps or {}

    # ------------------------------------------------------------------
    # 1. Similarity: cosine + mean normalization + sparse handling
    # ------------------------------------------------------------------
    def calculate_similarity(self, vec_a=None, vec_b=None):
        """Cosine similarity between two sparse rating/feature vectors.

        - Vectors are dicts {key: value}; missing keys are unknown and are
          NOT treated as 0 (sparse handling).
        - Values are mean-centered (normalization) so users who rate
          everything high are not "similar to everyone".
        - Only keys present in BOTH vectors contribute.
        """
        if vec_a is None or vec_b is None:
            users = list(self.ratings)
            if len(users) >= 2 and vec_a is None and vec_b is None:
                vec_a = self.ratings[users[0]]
                vec_b = self.ratings[users[1]]
            else:
                return 0.0

        common = set(vec_a) & set(vec_b)
        if not common:
            return 0.0

        mean_a = _mean(vec_a.values())
        mean_b = _mean(vec_b.values())

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for key in common:
            a = vec_a[key] - mean_a
            b = vec_b[key] - mean_b
            dot += a * b
            norm_a += a * a
            norm_b += b * b

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    # ------------------------------------------------------------------
    # 2. Collaborative filtering: user/item modes + neighbors + prediction
    # ------------------------------------------------------------------
    def collaborative_filtering(self, user_id=None, mode="user", k=10, n=5):
        """Predict ratings via user-based or item-based collaborative filtering.

        - mode="user": similar users (neighbors) vote on unseen items.
        - mode="item": scores come from similar items the user rated.
        - Only the top-k most similar neighbors are used.
        - Predictions are real rating estimates (baseline + weighted
          average of neighbor deviations), not raw similarity values.
        Returns a sorted list of (item_id, predicted_rating).
        """
        if not self.ratings:
            return []
        if user_id is None:
            user_id = next(iter(self.ratings))
        if user_id not in self.ratings:
            return self.handle_cold_start(user_id, n=n)

        if mode == "user":
            return self._user_based_cf(user_id, k, n)
        if mode == "item":
            return self._item_based_cf(user_id, k, n)
        raise ValueError(f"unknown mode: {mode!r} (expected 'user' or 'item')")

    def _user_based_cf(self, user_id, k, n):
        target = self.ratings[user_id]
        target_mean = _mean(target.values())

        neighbors = []
        for other_id, other_ratings in self.ratings.items():
            if other_id == user_id:
                continue
            sim = self.calculate_similarity(target, other_ratings)
            if sim > 0:
                neighbors.append((sim, other_id, other_ratings))
        # Neighbor selection: keep only the top-k most similar users.
        neighbors.sort(key=lambda x: x[0], reverse=True)
        neighbors = neighbors[:k]

        candidates = {}
        for sim, _, other_ratings in neighbors:
            other_mean = _mean(other_ratings.values())
            for item, rating in other_ratings.items():
                if item in target:
                    continue
                num, den = candidates.get(item, (0.0, 0.0))
                candidates[item] = (num + sim * (rating - other_mean),
                                    den + abs(sim))

        # Rating prediction: target baseline + weighted neighbor deviation.
        predictions = [
            (item, target_mean + num / den)
            for item, (num, den) in candidates.items() if den > 0
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    def _item_based_cf(self, user_id, k, n):
        target = self.ratings[user_id]
        target_mean = _mean(target.values())

        # Build item -> {user: rating} view.
        item_users = {}
        for uid, user_ratings in self.ratings.items():
            for item, rating in user_ratings.items():
                item_users.setdefault(item, {})[uid] = rating

        predictions = []
        for item in item_users:
            if item in target:
                continue
            # Neighbor selection: top-k items most similar to this one
            # among items the user already rated.
            neighbors = []
            for rated_item, rating in target.items():
                sim = self.calculate_similarity(item_users[item],
                                                item_users[rated_item])
                if sim > 0:
                    neighbors.append((sim, rated_item, rating))
            neighbors.sort(key=lambda x: x[0], reverse=True)
            neighbors = neighbors[:k]
            if not neighbors:
                continue
            num = sum(sim * rating for sim, _, rating in neighbors)
            den = sum(abs(sim) for sim, _, _ in neighbors)
            predictions.append((item, num / den if den else target_mean))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    # ------------------------------------------------------------------
    # 3. Content-based: features + weights + user profile
    # ------------------------------------------------------------------
    def content_based_recommend(self, user_id=None, n=5):
        """Recommend items by matching weighted item features against a
        learned user profile.

        - Items are described by feature vectors (not all identical).
        - Features are weighted by self.feature_weights.
        - The user profile is the rating-weighted average of the features
          of items the user has rated.
        Returns a sorted list of (item_id, score).
        """
        if not self.item_features:
            return []
        if user_id is None:
            user_id = next(iter(self.ratings), None)
        user_ratings = self.ratings.get(user_id, {})

        profile = self._build_user_profile(user_ratings)
        if profile is None:
            return self.handle_cold_start(user_id, n=n)

        scores = []
        for item, features in self.item_features.items():
            if item in user_ratings:
                continue
            score = self._weighted_cosine(profile, features)
            scores.append((item, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def _weight(self, feature):
        return self.feature_weights.get(feature, 1.0)

    def _build_user_profile(self, user_ratings):
        """User profile: rating-weighted average of item feature vectors."""
        profile = {}
        total = 0.0
        for item, rating in user_ratings.items():
            features = self.item_features.get(item)
            if not features:
                continue
            total += rating
            for feature, value in features.items():
                profile[feature] = profile.get(feature, 0.0) + rating * value
        if total <= 0 or not profile:
            return None
        return {f: v / total for f, v in profile.items()}

    def _weighted_cosine(self, profile, features):
        keys = set(profile) & set(features)
        if not keys:
            return 0.0
        dot = sum(self._weight(f) * profile[f] * features[f] for f in keys)
        norm_p = math.sqrt(sum(self._weight(f) * profile[f] ** 2 for f in profile))
        norm_i = math.sqrt(sum(self._weight(f) * features[f] ** 2 for f in features))
        if norm_p == 0.0 or norm_i == 0.0:
            return 0.0
        return dot / (norm_p * norm_i)

    # ------------------------------------------------------------------
    # 4. Cold start: strategy + popularity + exploration
    # ------------------------------------------------------------------
    def handle_cold_start(self, user_id=None, n=5, min_ratings=3,
                          exploration_ratio=0.2, seed=None):
        """Cold-start strategy.

        - Users with enough history get normal recommendations.
        - New users get popular items (most/best rated) so they always
          receive recommendations.
        - New/unseen items get an exploration boost so they can be
          discovered instead of never being shown.
        - A fraction of slots is reserved for exploration (random items)
          instead of always recommending the known top items.
        Returns a sorted list of (item_id, score).
        """
        rng = random.Random(seed)
        user_ratings = self.ratings.get(user_id, {})

        if len(user_ratings) >= min_ratings:
            return self.collaborative_filtering(user_id, n=n)

        # Popularity: average rating and rating count per item.
        count = {}
        total = {}
        for ratings in self.ratings.values():
            for item, rating in ratings.items():
                count[item] = count.get(item, 0) + 1
                total[item] = total.get(item, 0.0) + rating

        all_items = set(count) | set(self.item_features)
        scored = []
        for item in all_items:
            if item in user_ratings:
                continue
            avg = total.get(item, 0.0) / count[item] if item in count else 0.0
            # Bayesian-ish popularity plus an exploration boost for new
            # items (few or no ratings) so they are not invisible.
            popularity = avg * math.log1p(count.get(item, 0))
            novelty_boost = 1.0 / (1.0 + count.get(item, 0))
            scored.append((item, popularity + novelty_boost))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Exploration: reserve a fraction of slots for random items so we
        # do not always recommend the already-known popular ones.
        n_explore = int(n * exploration_ratio)
        n_exploit = n - n_explore
        picks = scored[:n_exploit]
        rest = [entry for entry in scored[n_exploit:]]
        if n_explore and rest:
            picks.extend(rng.sample(rest, min(n_explore, len(rest))))
        return picks[:n]

    # ------------------------------------------------------------------
    # 5. Ranking: relevance + diversity + business rules
    # ------------------------------------------------------------------
    def rank_recommendations(self, candidates=None, user_id=None, n=10,
                             diversity_penalty=0.3, ad_every=4, ads=None):
        """Rank candidate items.

        - Ranked by relevance score (predicted rating / content score),
          NOT by timestamp.
        - Diversity: items whose category already appears in the ranked
          list are penalized so same-category items do not cluster.
        - Business rules: ads are inserted at fixed positions
          (every `ad_every` slots).
        `candidates` is a list of (item_id, score); returns a ranked list
        of (item_id, score) with ads included as ("ad:<id>", score).
        """
        if candidates is None:
            if user_id is not None or self.ratings:
                uid = user_id if user_id is not None else next(iter(self.ratings))
                candidates = self.collaborative_filtering(uid, n=n * 2)
                if not candidates:
                    candidates = self.handle_cold_start(uid, n=n * 2)
            else:
                candidates = []

        # Relevance: sort by score descending (never by time).
        remaining = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Diversity: greedy re-ranking with a category-repetition penalty.
        ranked = []
        seen_categories = {}
        while remaining and len(ranked) < n:
            best_idx = 0
            best_adjusted = None
            for idx, (item, score) in enumerate(remaining):
                category = self.item_categories.get(item)
                repeats = seen_categories.get(category, 0) if category else 0
                adjusted = score - diversity_penalty * repeats
                if best_adjusted is None or adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_idx = idx
            item, score = remaining.pop(best_idx)
            ranked.append((item, score))
            category = self.item_categories.get(item)
            if category:
                seen_categories[category] = seen_categories.get(category, 0) + 1

        # Business rules: insert ads at fixed positions.
        if ads and ad_every > 0:
            ad_list = list(ads)
            ad_idx = 0
            position = ad_every
            while position <= len(ranked) and ad_idx < len(ad_list):
                ranked.insert(position, (f"ad:{ad_list[ad_idx]}", 0.0))
                ad_idx += 1
                position += ad_every + 1
        return ranked
