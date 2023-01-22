from modelling.model import content_recommender
from app.data_processors import AnimeSimilarityMatrix

anime_sm = AnimeSimilarityMatrix()

class AnimeContentRecommender(content_recommender.ContentRecommender):
    """
    A class for building an Anime Content-Based Recommender
    """
    def __init__(
        self,
        dataframe_csv_location: str = 'modelling/data/df.csv',
        item_name_column: str = 'Name',
        similarity_matrix = anime_sm.sm,
        set_threshold: int = 0.3
    ):
        """
        Inherits from ContentRecommender class.

        :param dataframe_csv_location:
            Location of the dataframe to search through for anime recommendations
        :param item_name_column:
            Column name for the anime name column
        :param similarity_matrix_location:
            Location of the numpy matrix of similarity scores
        :param set_threshold:
            The threshold for the similarity scores
        """
        super().__init__(
            dataframe_csv_location=dataframe_csv_location,
            item_name_column=item_name_column,
            similarity_matrix=similarity_matrix,
            set_threshold=set_threshold
        )