import numpy as np
import pandas as pd

class ContentRecommender():
    def __init__(
        self,
        dataframe_csv_location: str,
        item_name_column: str,
        similarity_matrix,
        set_threshold: int
    ):
        """
        Class for the Content-Based Recommender

        :param dataframe_csv_location:
            Location of the dataframe to search through for recommendations
        :param item_name_column:
            Column name for the item name column
        :param similarity_matrix_location:
            Location of the numpy matrix of similarity scores
        :param set_threshold:
            The threshold for the similarity scores
        """
        self.df = pd.read_csv(dataframe_csv_location)
        self.col = item_name_column
        self.sm = similarity_matrix
        self.threshold = set_threshold
        
    def get_recommendations(
        self,
        item_name: str,
        sort_by: str = 'Similarity_Scores',
        recommendation_amount: int = 10,
        threshold: int = None
    ):
        """
        Method to get recommendations

        :param item_name:
            Name of the item to get recommendations similar to
        :param sort_by:
            The column to sort the recommendations by (default: Similarity Scores)
        :param sort_by_ascending:
            To sort by ascending or not for the sort_by column (default: False)
        :param threshold:
            The threshold for the similarity scores. If not specified, initialised default.
        :param recommendation_amount:
            How many recommendations to return (default: 10)
        """
        # Set threshold
        if not threshold:
            threshold = self.threshold

        # Collect errors
        errors = []

        # Error for not found items
        if item_name not in self.df[self.col].values:
            errors.append("Could not find anything by that name. Please check punctuation and spelling.")

        # Error for threshold not between 0 and 1
        if threshold < 0 or threshold > 1:
            errors.append("Please specify a threshold between 0 and 1.")

        # Return errors if any
        if errors:
            return errors

        # Index of item
        item_idx = self.df.index[self.df[self.col] == item_name].to_list()[0]

        # Index of similar items
        sim_idx = np.where(self.sm[item_idx] > threshold)[1].tolist()

        # Find similarity scores of similar items
        sim_scores = np.array(self.sm[item_idx].tolist()[0])[sim_idx].tolist()

        # Get recommendations
        recoms = self.df.iloc[sim_idx]
        
        # Add Similarity Scores column
        recoms['Similarity_Scores'] = sim_scores

        # Bring back only total
        if recommendation_amount > len(recoms):
            recommendation_amount = len(recoms)

        # Return
        return recoms.sort_values(by=sort_by, ascending=False).head(recommendation_amount)