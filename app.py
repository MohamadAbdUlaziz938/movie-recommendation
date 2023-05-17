from apriori_algorithm import Apriori
from data_processing import build_matrix, process_data
from item_based_collaborative import ItemBasedCollaborative
from popular import MostPopularRating, PopularRecommendation
from read_data import load_data

if __name__ == "__main__":
    data = load_data(movie_path="data/movies.csv", ratings_path="data/ratings.csv")
    process_data(data=data)
    print(data)
    data_matrix = build_matrix(data=data)
    print(data_matrix)
    '''Item based recommendation'''
    print("Item based recommendation")
    item_based_collaborative = ItemBasedCollaborative(dataRecommendation=data_matrix)
    item_based_collaborative.fit_model()
    results = item_based_collaborative.movie_recommendation(
        movie_name="Batman Returns", num_of_recommendations=7
    )
    print(results)

    '''Aprior algorith'''
    print("\nAprior\n")
    aprior = Apriori(data=data_matrix)
    aprior.fit()
    aprior.recomend()

    '''Popularity recommendation'''
    print('Most ten popular recommendation')
    mostPopularRating=MostPopularRating(data=data)
    results=mostPopularRating.get_most_popular()
    print(results)

    print('Most popular recommendation')
    popularRating=PopularRecommendation(train_data=data)
    popularRating.create()
    resutls=popularRating.recommend(user_id="23dadwA",num_recommendations=10)
    print(results)