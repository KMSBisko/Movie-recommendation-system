import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
file_ratings_path = 'data/movies/ratings.csv'
ratings = pd.read_csv(file_ratings_path)

# Split the data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.3, random_state=0)

# Create user-item matrices for training and testing
train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
test_matrix = test_data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

# Normalize Ratings
def normalize_ratings(matrix):
    user_mean = matrix.mean(axis=1)
    normalized_matrix = matrix.sub(user_mean, axis=0)
    return normalized_matrix

normalized_train_matrix = normalize_ratings(train_matrix)

# Calculate the cosine similarity between users using the normalized matrix
user_similarity = cosine_similarity(normalized_train_matrix, normalized_train_matrix)

# Predict ratings for the test set using the normalized matrix
def predict_ratings(user_similarity, train_matrix):
    mean_user_rating = train_matrix.mean(axis=1)
    ratings_diff = (train_matrix.to_numpy() -
                    mean_user_rating.to_numpy()[:, np.newaxis])
    pred = mean_user_rating.to_numpy()[:, np.newaxis] + user_similarity.dot(
        ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return pred

user_predictions = predict_ratings(user_similarity, normalized_train_matrix)
user_predictions_df = pd.DataFrame(user_predictions, columns=train_matrix.columns, index=train_matrix.index)

# Evaluate the model using Root Mean Squared Error (RMSE)
def rmse(pred, actual):
    non_zero_indices = actual.nonzero()
    pred_flatten = pred[non_zero_indices]
    actual_flatten = actual[non_zero_indices]
    mse = mean_squared_error(pred_flatten, actual_flatten)
    return sqrt(mse)

rmse_value = rmse(user_predictions, test_matrix.values)
print(f'RMSE: {rmse_value}')

# Recommend movies for a specific user
def get_top_recommendations(user_predictions_df, userID, top_n=10):
    user_predictions = user_predictions_df.loc[userID].sort_values(ascending=False)
    top_n_recommendations = user_predictions.head(top_n)
    return top_n_recommendations

# Example usage:
user_input = input("Enter userID to predict (enter 0 to stop): ")
userID_to_predict = int(user_input)
top_recommendations = get_top_recommendations(user_predictions_df, userID_to_predict)
print(f'Top recommendations for userID {userID_to_predict}:\n{top_recommendations}')
