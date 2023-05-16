import pandas as pd
def get_matrix(movie_path:str,ratings_path:str):
    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(ratings_path)
    data = pd.merge(ratings, movies, how='inner')
    
    return data