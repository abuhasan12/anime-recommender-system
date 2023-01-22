import numpy as np
from scipy import sparse
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import StopWordsRemover
from sklearn.feature_extraction.text import TfidfVectorizer

class SimilarityMatrix():
    """
    Used to generate a similarity matrix from a csv file.
    The csv file location is passed as an argument to the class constructor.
    """
    def __init__(
        self,
        df_csv_location: str
    ):
        """
        Initializes the class by creating a spark session and reading the csv file specified in the location passed as an argument.
        
        :param df_csv_location:
            Path of csv file
        """
        spark = SparkSession.Builder().appName("AnimeRecommender").getOrCreate()
        self.sdf = spark.read.csv(df_csv_location, header=True)
    
    def drop_col(
        self,
        col
    ):
        """
        Drops the specified column from the dataframe.
        
        :param col
            Name of the column to be dropped
        """
        self.sdf = self.sdf.drop(col)
    
    def one_hot_encode(
        self,
        col: str
    ):
        """
        One hot encodes the specified column.

        :param col:
            Column to perform one hot encoding on.
        """
        self.sdf = self.sdf.withColumn('Col_split', F.split(self.sdf[col], ', '))
        col_set = self.sdf.withColumn('exploded_col', F.explode('Col_split')).agg(F.collect_set('exploded_col')).collect()[0][0]
        col_set = sorted(col_set)
        for c in col_set:
            self.sdf = self.sdf.withColumn(c, F.when(F.array_contains('Col_split', c), 1).otherwise(0))
        self.sdf = self.sdf.drop(col, 'Col_split') # remember to drop 6.45 and 7.14
    
    def clean_text(
        self,
        col: str
    ):
        """
        Cleans the text in the specified column by removing special characters, lowercasing, removing stopwords and concatenating the remaining words.

        :param col:
            Column to perform operations on
        """
        cleaning_col = col + "_cleaning"
        cleaned_col = col + "_clean"
        self.sdf = self.sdf.withColumn(cleaning_col, F.regexp_replace(col, r"""[!\"#$%&'()*+,\-.\/:;<=>?@\[\\\]^_`{|}~]""", ''))
        self.sdf = self.sdf.withColumn(cleaning_col, F.lower(F.col(cleaning_col)))
        self.sdf = self.sdf.withColumn(cleaning_col, F.split(self.sdf[cleaning_col], ' '))
        remover = StopWordsRemover().setInputCol(cleaning_col).setOutputCol(cleaned_col)
        self.sdf = remover.transform(self.sdf)
        self.sdf = self.sdf.withColumn(cleaned_col, F.concat_ws(" ",F.col(cleaned_col)))
        self.sdf = self.sdf.drop(col, cleaning_col)

    def to_pandas(
        self
    ):
        """
        Converts the spark dataframe to pandas dataframe
        """
        self.df = self.sdf.toPandas()
    
    def sm_from_columns(
        self,
        include: list = None,
        exclude: list = None
    ):
        """
        Creates a similarity matrix from the columns specified in the include or exclude list.

        :param include:
            list of columns to be included for similarity matrix generation.
        :param exclude:
            list of columns to be excluded for similarity matrix 
        """
        if include:
            df = self.df[include].copy()
        elif exclude:
            df = self.df.drop(exclude, axis=1)
        arr = df.to_numpy()
        matr = np.dot(arr, np.transpose(arr))
        max_vec = matr.max(axis=1)
        min_vec = matr.min(axis=1)
        max_min_vec = max_vec - min_vec
        norm_matr = (matr - min_vec[:,None])/(max_min_vec[:,None])
        self.sm_from_cols = sparse.csc_matrix(norm_matr)
    
    def sm_from_text_col(
        self,
        col: str
    ):
        """
        Creates a similarity matrix from the text in the specified column

        :param col:
            name of the column containing the text
        """
        v = TfidfVectorizer()
        tfidf = v.fit_transform(self.df[col])
        self.sm_from_text = np.dot(tfidf, np.transpose(tfidf))
    
    def combine_sms(
        self
    ):
        """
        Combines the similarity matrices generated from columns and text to create a final similarity matrix.
        """
        av_sm = (self.sm_from_cols + self.sm_from_text) / 2
        gmean_sm = self.sm_from_cols.multiply(self.sm_from_text).sqrt()
        av_gmean_sm = (av_sm + gmean_sm) / 2
        self.sm = av_gmean_sm.todense()