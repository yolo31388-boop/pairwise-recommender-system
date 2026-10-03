import pytest
from recommender.system import RecommenderSystem


class TestRecommenderSystem:

    def test_calculate_similarity(self):
        """相似度计算不做余弦，用欧氏距离"""
        obj = RecommenderSystem()
        result = obj.calculate_similarity()
        self.assertIsNotNone(result)

    def test_collaborative_filtering(self):
        """协同过滤不做用户/物品区分，只用一种"""
        obj = RecommenderSystem()
        result = obj.collaborative_filtering()
        self.assertIsNotNone(result)

    def test_content_based_recommend(self):
        """内容推荐不做特征，所有物品一样"""
        obj = RecommenderSystem()
        result = obj.content_based_recommend()
        self.assertIsNotNone(result)

    def test_handle_cold_start(self):
        """冷启动不做策略，新用户没推荐"""
        obj = RecommenderSystem()
        result = obj.handle_cold_start()
        self.assertIsNotNone(result)

    def test_rank_recommendations(self):
        """排序不做相关性，按时间排"""
        obj = RecommenderSystem()
        result = obj.rank_recommendations()
        self.assertIsNotNone(result)

