import numpy as np
import pandas as pd
from surprise import SVD
from surprise import Reader, Dataset
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Read the CSV file into a DataFrame, excluding the index column
file_path = 'data/movies/ratings.csv'
train_data = pd.read_csv(file_path).set_index('userId')

# Drop the 'timestamp' column if you don't want to use it
train_data = train_data.drop(columns=['timestamp'])
print(train_data)
# Assuming you have the following variables defined
# Replace them with your actual data
user = 1
movie = 1
final_data = pd.DataFrame()
x_test = pd.DataFrame()
y_test = pd.Series()
y_true = pd.Series()

# Split the data into training and testing sets
train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Specify how to read the data frame.
reader = Reader(rating_scale=(1, 5))

# Create the traindata from the data frame.
train_data_mf = Dataset.load_from_df(train_data[['movieId', 'rating']], reader)

# Build the train set from traindata.
trainset = train_data_mf.build_full_trainset()

# Initialize and fit SVD model.
svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
svd.fit(trainset)

# Getting predictions of the train set.
train_preds = svd.test(trainset.build_testset())
train_pred_mf = np.array([pred.est for pred in train_preds])

# Creating a sparse matrix.
train_sparse_matrix = csr_matrix((train_data.rating.values, (train_data.index, train_data.movieId.values)))

train_averages = dict()

# Get the global average of ratings in our train set.
train_global_average = train_sparse_matrix.sum() / train_sparse_matrix.count_nonzero()
train_averages['global'] = train_global_average

# ... (Rest of your code)

# Assuming your variables are correctly defined, the code should work.
# If you still encounter issues, please check for typos and ensure all necessary libraries are installed.

# ... (Previous code)

# Next, let’s create a function which takes the sparse matrix as input and gives the average rating of a movie given by all users,
# and the average rating of all movies given by a single user.


def get_average_rating(sparse_matrix, of_users):
    # Average rating of user/axes.
    ax = 1 if of_users else 0  # 1 - User axes, 0 - Movie axes.
    # ".A1" is for converting Column_Matrix to 1-D numpy array.
    sum_of_rating = sparse_matrix.sum(axis=ax).A1
    # Boolean matrix of rating (whether a user rated that movie or not).
    is_rated = sparse_matrix != 0
    # No of rating that each user OR movie.
    no_of_rating = is_rated.sum(axis=ax).A1
    # Max_user and max_movie ids in sparse matrix.
    u, m = sparse_matrix.shape
    # Create a dictionary of users and their average rating.
    average_rating = {i: sum_of_rating[i] / no_of_rating[i]
                       for i in range(u if of_users else m) if no_of_rating[i] != 0}
    # Return that dictionary of average rating.
    return average_rating


train_averages['user'] = get_average_rating(
    train_sparse_matrix, of_users=True)
train_averages['movie'] = get_average_rating(
    train_sparse_matrix, of_users=False)

# Compute the similar movies of the "movie".
movie_sim = cosine_similarity(
    train_sparse_matrix[:, movie].T, train_sparse_matrix.T).ravel()
top_sim_movies = movie_sim.argsort()[::-1][1:]

# We are ignoring 'The User' from its similar users.
# Get the rating of the most similar movie rated by this user.
top_rating = train_sparse_matrix[user, top_sim_movies].toarray().ravel()

# We will make its length "5" by adding user averages to it.
top_sim_movies_rating = list(top_rating[top_rating != 0][:5])
top_sim_movies_rating.extend(
    [train_averages['user'][user]] * (5 - len(top_sim_movies_rating)))

# Prepare train data.
x_train = final_data.drop(['userID', 'movieID', 'rating'], axis=1)
y_train = final_data['rating']

# Initialize XGBoost model.
xgb_model = xgb.XGBRegressor(
    silent=False, n_jobs=13, random_state=15, n_estimators=100)

# Fit the model.
xgb_model.fit(x_train, y_train, eval_metric='rmse')

# Dictionaries for storing train and test results.
test_results = dict()

# From the trained model, get the predictions.
y_test_pred = xgb_model.predict(x_test)

# Get the RMSE and MAPE of the test data.
rmse_test = np.sqrt(np.mean((y_test.values - y_test_pred)**2))
mape_test = np.mean(
    np.abs((y_test.values - y_test_pred) / y_true.values)) * 100

# Store the results in test_results dictionary.
test_results = {'rmse': rmse_test,
                'mape': mape_test, 'predictions': y_test_pred}

# Assuming your variables are correctly defined, the code should work.
# If you still encounter issues, please check for typos and ensure all necessary libraries are installed.
