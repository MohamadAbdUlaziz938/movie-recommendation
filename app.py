from aprior_algorithm import Aprior
from data_processing import build_matrix, process_data
from item_based_collaborative import ItemBasedCollaborative
from read_data import load_data

if __name__ == "__main__":
    data = load_data(movie_path="data/movies.csv", ratings_path="data/ratings.csv")
    print(data)
    process_data(data=data)
    data_matrix = build_matrix(data=data)
    print(data_matrix)

    print("Item based recommendation")
    item_based_collaborative = ItemBasedCollaborative(dataRecommendation=data_matrix)
    item_based_collaborative.fit_model()
    results = item_based_collaborative.movie_recommendation(
        movie_name="Batman Returns", num_of_recommendations=7
    )
    print(results)

    # Aprior algorith
    print("\nAprior\n")
    aprior = Aprior(data=data_matrix)
    aprior.fit()
    aprior.recomend()
