from modelling.data_processor import similarity_matrix

class AnimeSimilarityMatrix(similarity_matrix.SimilarityMatrix):
    """
    Class for building similarity matrix for Anime Recommender
    """
    def __init__(
        self,
        df_csv_location: str = 'modelling/data/df.csv'
    ):
        """
        Inherits from SimilarityMatrix class and performs operations to build similarity matrix

        :param df_csv_location:
            Location of the dataframe csv.
        """
        super().__init__(df_csv_location=df_csv_location)
        self.one_hot_encode(col='Genres')
        self.drop_col(col='6.45')
        self.drop_col(col='7.14')
        self.clean_text(col='Synopsis')
        self.to_pandas()
        self.sm_from_columns(exclude=['Name', 'Score', 'Synopsis_clean'])
        self.sm_from_text_col(col='Synopsis_clean')
        self.combine_sms()