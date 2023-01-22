import unittest
from modelling.model.content_recommender import ContentRecommender
from app.data_processors import AnimeSimilarityMatrix

class TestContentRecommender(unittest.TestCase):
    def setUp(self):
        """
        Assign a test dataframe and similarity matrix to use for testing
        """
        anime_sm = AnimeSimilarityMatrix()
        self.cr = ContentRecommender(
            dataframe_csv_location='modelling/data/df.csv',
            item_name_column='Name',
            similarity_matrix=anime_sm.sm,
            set_threshold=0.3
        )
        
    def test_init(self):
        """
        Test initializing the class with correct inputs
        """
        self.assertIsInstance(self.cr, ContentRecommender)
        
    def test_get_recommendations(self):
        """
        Test getting recommendations with correct inputs
        """
        recommendations = self.cr.get_recommendations(item_name='Cowboy Bebop')
        self.assertEqual(len(recommendations) > 1, True)
        
    def test_get_recommendations_threshold(self):
        """
        Test getting recommendations with a different threshold
        """
        recommendations = self.cr.get_recommendations(item_name='Cowboy Bebop', threshold=0.99)
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations['Name'].tolist(), ['Cowboy Bebop'])
        
    def test_get_recommendations_errors(self):
        """
        Test getting recommendations with incorrect inputs
        """
        recommendations = self.cr.get_recommendations(item_name='Not an anime')
        self.assertEqual(recommendations, ["Could not find anything by that name. Please check punctuation and spelling."])
        
        recommendations = self.cr.get_recommendations(item_name='Cowboy Bebop', threshold=2)
        self.assertEqual(recommendations, ["Please specify a threshold between 0 and 1."])