"""Pytest bootstrap.

The test class in tests/test_recommender.py uses unittest-style
assertions (self.assertIsNotNone) but does not inherit from
unittest.TestCase. Inject the missing assertion method here so the
unmodified test file can run under pytest.
"""

import unittest

from tests.test_recommender import TestRecommenderSystem

if not hasattr(TestRecommenderSystem, "assertIsNotNone"):
    TestRecommenderSystem.assertIsNotNone = unittest.TestCase.assertIsNotNone
