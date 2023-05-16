import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


class ItemBasedCollaborative:
    def __init__(self, dataRecommendation: pd.DataFrame) -> None:
        self.knn = None
        self.dataRecommendation = dataRecommendation.copy()

    def fit_model(self):
        self.knn = NearestNeighbors(
            n_neighbors=11, metric="cosine", algorithm="brute", n_jobs=-1
        )
        self.knn.fit(self.dataRecommendation.values.T)

    def movie_recommendation(self, movie_name, num_of_recommendations):
        a = (
            self.dataRecommendation.columns.to_frame()
            .reset_index(drop=True)
            .to_dict()["movieId"]
        )
        recommendation_result = list(
            self.knn.kneighbors(
                [self.dataRecommendation[movie_name].values], num_of_recommendations + 1
            )
        )
        recommendation_result = pd.DataFrame(
            np.vstack((recommendation_result[1], recommendation_result[0])),
            index=["movieId", "Cosine_Similarity (degree)"],
        ).T
        recommendation_result = recommendation_result.drop([0]).reset_index(drop=True)
        recommendation_result.movieId = recommendation_result.movieId.map(a)
        return recommendation_result
