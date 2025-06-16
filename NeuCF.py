#from Utils import NeuCF
from DataPreprocessing import *
#import matplotlib.pyplot as plt
import numpy as np
import keras

x = ratings[['user', 'movie']].values
y = ratings['user_rate'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# 生成训练集和测试集
n_user = len(ratings['userID'].unique())
n_item = len(ratings['movieID'].unique())


model = keras.models.load_model('model.keras')

# model = NeuCF(n_user, n_item, dim=50, l2=1e-6)
# if not os.path.exists('model.keras'):
#     history = model.fit([x[:, 0], x[:, 1]], y, epochs=10, batch_size=256, validation_split=0.2)
#     model.save('model.keras')
#     # 模型评估
#     plt.plot(history.history['loss'], label='loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend()
#     plt.show()
# else:
#     model = keras.models.load_model('model.keras')

# print([x[:, 0], x[:, 1]])
# history = model.fit([x[:, 0], x[:, 1]], y, epochs=10, batch_size=256, validation_split=0.2)

user_id = 25
movies_watched_by_user = ratings[ratings.userID == user_id]
movies_not_watched = ratings[
    ~ratings["movieID"].isin(movies_watched_by_user.movieID.values)
]["movieID"]
movies_not_watched = list(
    set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
)
movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
user_encoder = user2user_encoded.get(user_id)

user_movie_array = [[], []]
for movie in movies_not_watched:
    user_movie_array[0].append(user_encoder)
    user_movie_array[1].append(movie[0])

user_movie_array = np.array(user_movie_array)

rating = model.predict([user_movie_array[0], user_movie_array[1]]).flatten()
top_ratings_indices = rating.argsort()[-16:][::-1]
recommended_movie_ids = [
    movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
]

top_movies_user = (
    movies_watched_by_user.sort_values(by='user_rate', ascending=False)
    .head(10)
    .movieID.values
)

# 打印推荐结果
print("Movies this user has watched:")
print("----" * 8)
movies_watched_by_user = movieInfo[movieInfo["movieID"].isin(movies_watched_by_user.movieID.values)]
for row in movies_watched_by_user.itertuples():
    print(row.name, ":", row.genre)
print("----" * 8)

print("Showing recommendations for user: {}".format(user_id))
print("====" * 9)
print("Movies with high ratings from user")
print("----" * 8)

movie_df_rows = movieInfo[movieInfo["movieID"].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.name, ":", row.genre)

print("----" * 8)
print("Top 16 movie recommendations")
print("----" * 8)
recommended_movies = movieInfo[movieInfo["movieID"].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.name, ":", row.genre)

# loss, mean_squared_error = model.evaluate([x[:, 0], x[:, 1]], y, verbose=1)
#
# print("mean_squared_error: %.2f" % (mean_squared_error))

def get_movie_recommendations(user_id, connection):

    # 处理出未观看电影
    movies_watched_by_user = ratings[ratings.userID == user_id]
    movies_not_watched = ratings[
        ~ratings["movieID"].isin(movies_watched_by_user.movieID.values)
    ]["movieID"]
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)

    if user_encoder is None:
        print(f"Invalid user ID: {user_id}")
        return []

    # 处理出输入数组
    user_movie_array = [[], []]
    for movie in movies_not_watched:
        user_movie_array[0].append(user_encoder)
        user_movie_array[1].append(movie[0])

    user_movie_array = np.array(user_movie_array)

    #获取用户对所有未观看电影的评分
    rating = model.predict([user_movie_array[0], user_movie_array[1]]).flatten()
    valid_indices = ~np.isnan(rating)  # 获取非NaN值的布尔索引
    valid_ratings = rating[valid_indices]  # 筛选出非NaN的评分

    #推荐十六部预测分数最高的电影
    top_ratings_indices = valid_ratings.argsort()[-16:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    recommended_movies = movieInfo[movieInfo["movieID"].isin(recommended_movie_ids)]
    for row in recommended_movies.itertuples():
        print(row.name, ":", row.genre)

    # 将结果转换为字典格式
    result = []
    for movie in recommended_movie_ids:
        result.append({
            'id': movie
        })

    return result

# 预处理所有用户的推荐电影并存入数据库
def store_all_recommendations():
    connection = sqlite3.connect('movieInfo.db')
    cursor = connection.cursor()

    for user_id in range(1, 3328):  # 用户ID从1到3327
        result = get_movie_recommendations(user_id, connection)
        # 存入推荐电影
        query = '''
            INSERT INTO user_recommendations (user_id, movie_id)
            VALUES (?, ?)
        '''
        for movie in result:
            cursor.execute(query, (
                user_id,
                int(movie['id'])
            ))

    connection.commit()
    connection.close()

#store_all_recommendations()

# connection = sqlite3.connect('movieInfo.db')
# result = get_movie_recommendations(25, connection)