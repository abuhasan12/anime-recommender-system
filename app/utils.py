def get_recommendations(
    anime: str,
    sort: str,
    n_recoms: int,
    threshold: int,
    recommender
):
    """
    Gets recommendations for anime input.

    :param anime:
        The anime to get a recommendation for
    :param sort:
        The column to sort by
    :param ascending:
        If the sort should be ascending or not
    :param n_recoms:
        Number of recommendations to return
    :param ascending:
        Threshold for similarity scores
    :param recommender:
        Built recommender to use
    """
    response = recommender.get_recommendations(anime, sort, n_recoms, threshold)
    return response