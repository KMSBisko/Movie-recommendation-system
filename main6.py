import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_ratings_path = "data/movies/ratings.csv"
rating = pd.read_csv(file_ratings_path)
df=rating

def create_user_movie_df(dataframe):
    # comment
    comment_counts = pd.DataFrame(dataframe["movieId"].value_counts())
    # print(comment_counts.head(20))

    # rare movies
    rare_movies = comment_counts[comment_counts["movieId"] >= 1000].index
    # print(rare_movies)
    
    common_movies = df[~df["movieId"].isin(rare_movies)]
    # common_movies = df
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["movieId"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df(rating)

def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    # Chọn phim mà người dùng đã xem
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    print(f"Số lượng bộ phim mà người dùng ID: [{random_user}] đã xem là: {len(movies_watched)}")

    # ===== Truy cập dữ liệu và id của những người dùng khác đang xem cùng một bộ phim ===== #
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    # ===== Chọn những người dùng khác đã xem cùng một bộ phim ===== #
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
    # print(users_same_movies)

    # ===== Xác định người dùng có hành vi giống nhất với người dùng để được đề xuất ===== #
    # Để làm điều này, chúng ta sẽ thực hiện 3 bước:
    #   Chúng tôi sẽ thu thập dữ liệu của Người dùng của chúng tôi và những người dùng khác.
    #   Chúng ta sẽ tạo mối tương quan df.
    #   Chúng tôi sẽ tìm những người dùng tương tự nhất (Top Users)

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])
    # print(final_df.head())
    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    # print(corr_df)
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()
    # print(corr_df.head)
    # print(corr_df)
    # print(corr_df[(corr_df["user_id_1" ] == random_user) & (corr_df["corr"] >= cor_th)])
    
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True) # select similar users have correlation over cor_th on random_user
    # print(top_users)
    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv(file_ratings_path)
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    # ===== Tính điểm đề xuất trung bình có trọng số ===== #
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
    # print(top_users_ratings.head())
    # print(top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"}).head())
    
    # ===== Thuật toán đề xuất film ===== #
    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()
    # print(recommendation_df)
    # print(recommendation_df[recommendation_df["weighted_rating"] > 3.5])
    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)

    return movies_to_be_recommend

def ShowAnswer(UserID = 0, cor_th = 0.70, score = 4):
  print("=============== Thông tin ===============")
  print("                                         ")
  print(f"Users ID: \t {UserID}")
  print(f"Score : \t {score}")
  print(f"cor_th: \t {cor_th}")
  print("                                         ")
  print("=========== Phim được đề xuất ===========")
  print("                                         ")
  print(user_based_recommender(UserID, user_movie_df, cor_th=cor_th, score=score))

# @markdown ---
# @markdown ### Nhập dữ liệu:
random_user = 1231  # @param {type:"number"}
score = 4 # @param {type:"slider", min:1, m67ax:5, step:0.5}
cor_th = 0.9 # @param {type:"slider", min:0.5, max:1, step:0.1}
if random_user == 0:
  random_user = int(pd.Series(user_movie_df.index).sample(1).values)
# @markdown ---

ShowAnswer(UserID=random_user, cor_th=cor_th, score=score)