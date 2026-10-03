"""
pairwise-recommender-system - RecommenderSystem

This module contains a deliberately broken implementation.
Fix all bugs so that tests pass.
"""


class RecommenderSystem:
    def __init__(self):
        self.state = {}

    def calculate_similarity(self, *args, **kwargs):
        """BUG: 相似度计算不做余弦，用欧氏距离"""
        return None

    def collaborative_filtering(self, *args, **kwargs):
        """BUG: 协同过滤不做用户/物品区分，只用一种"""
        return None

    def content_based_recommend(self, *args, **kwargs):
        """BUG: 内容推荐不做特征，所有物品一样"""
        return None

    def handle_cold_start(self, *args, **kwargs):
        """BUG: 冷启动不做策略，新用户没推荐"""
        return None

    def rank_recommendations(self, *args, **kwargs):
        """BUG: 排序不做相关性，按时间排"""
        return None

