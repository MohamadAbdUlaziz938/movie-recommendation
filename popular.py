import numpy as np
import pandas as pd

number_of_movie_ratings = "number_of_movie_ratings"
average_movie_rating="average_movie_rating"


# most popularity recommendation model
class PopularRecommendation:
    def __init__(self, train_data):
        self.train_data = train_data
        self.filtered = None
        self.min_number_rating = None
        self.average_rating = None

    def weighted_score(self, x):
        v = x[number_of_movie_ratings]
        R = x[average_movie_rating]
        # Calculation based on an IMDB formula
        return (v / (v + self.min_number_rating) * R) + (self.min_number_rating / (self.min_number_rating + v) * self.average_rating)

    def create(self):
        self.train_data[number_of_movie_ratings] = (
            self.train_data["movieId"]
            .groupby(self.train_data["movieId"])
            .transform("count")
        )
        # Create a column for the average rating a movie has received called 'average_rating'
        self.train_data[average_movie_rating] = (
            self.train_data["rating"]
            .groupby(self.train_data["movieId"])
            .transform("mean")
        )
        # Calculate the average rating for all movies
        self.average_rating = self.train_data["rating"].mean()
        # Calculate the minimum number of movie ratings needs to receive in order to be included in the model
        self.min_number_rating = self.train_data[number_of_movie_ratings].quantile(0.90)
        
        # Filter the dataset based on value m
        self.filtered = self.train_data.copy().loc[
            self.train_data[number_of_movie_ratings] >= self.min_number_rating
        ]
        # Create a 'score' column and give each movie a weighted score
        self.filtered["score"] = self.filtered.apply(self.weighted_score, axis=1)
        
        return self.filtered

    def recommend(self, user_id, num_recommendations=5):
        most_popular = self.filtered.sort_values("score", ascending=False)
        most_popular = most_popular.drop_duplicates(subset="movieId", keep="first")
        print("")
        print("popular recommendation")
        print("most popular for user_id {0}.".format(user_id))
        return most_popular[
            [
                "movieId",
                number_of_movie_ratings,
                average_movie_rating,
                "score",
            ]
        ].head(num_recommendations)


class MostPopularRating:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def get_most_popular(self):
        self.data[number_of_movie_ratings] = (
            self.data["movieId"].groupby(self.data["movieId"]).transform("count")
        )
        popular = self.data.sort_values(number_of_movie_ratings, ascending=False)
        popular = popular.drop_duplicates(subset="movieId", keep="first")
        popular = popular[["movieId", number_of_movie_ratings]]
        return popular.head(10)
