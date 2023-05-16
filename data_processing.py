import pandas as pd
import numpy as np


def process_data(data: pd.DataFrame):
    data["movie_year"] = data.title.str.extract(".*\((.*)\).*")
    data["title"] = data.title.str.split("(").str[0].str[:-1]

    movieFrequency_greater_10 = (
        data["movieId"].value_counts()[data["movieId"].value_counts() >= 10].index
    )
    data = data[data.movieId.isin(movieFrequency_greater_10)]
    return data


def build_matrix(data: pd.DataFrame):
    movieId_dict = (
        data.drop_duplicates("title")[["movieId", "title"]]
        .set_index("movieId")
        .to_dict()["title"]
    )

    dataRecommendation = data.pivot(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)

    dataRecommendation.columns = dataRecommendation.columns.map(movieId_dict)
    return dataRecommendation
