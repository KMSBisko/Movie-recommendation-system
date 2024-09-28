import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv("data/movies/ratings.csv")

final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
final_dataset.fillna(0, inplace=True)

sample = np.array([[0, 0, 3, 0, 0], [4, 0, 0, 0, 2], [0, 0, 0, 0, 1]])
sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
csr_sample = csr_matrix(sample)

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_movie_recommendation(movie_id):
    n_movies_to_recommend = 10
    # Assuming movies dataset contains a column 'movieId'
    movie_idx = final_dataset[final_dataset.index == int(movie_id)].index[0]
    distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)
    rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    recommend_frame = []
    for val in rec_movie_indices:
        # Assuming 'movieId' is the index of the final_dataset DataFrame
        recommend_frame.append({'movieId': final_dataset.iloc[val[0]].name, 'Distance': val[1]})
    df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend + 1))
    return df

# Example usage
while True:
    user_input = input("Enter a user ID (enter 0 to exit): ")
    # Check if the input is 0, if yes, exit the loop
    if user_input == '0':
        break
    try:
        user_id = int(user_input)
        # Call the function to get recommendations for the entered user ID
        recommendations = get_movie_recommendation(user_id)
        # Display the recommendations
        print(f"\nRecommendations for User ID {user_id}:\n")
        print(recommendations)
        
    except ValueError:
        print("Invalid input. Please enter a valid user ID or 0 to exit.")
