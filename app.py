from read_data import get_matrix

if __name__=="__main__":

 data= get_matrix(movie_path="data/movies.csv",ratings_path="data/ratings.csv")
 print(data)
