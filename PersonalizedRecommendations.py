import math
import random

import keras

from DataPreprocessing import *
import numpy as np

Tag = movieInfo['genre'].str.split(',', expand=True).stack().str.strip().unique()
Tag = Tag[Tag != '']
Movies = ratings['movieID'].unique()
MoviesEncoded = list(map(lambda x: movie2movie_encoded.get(x), Movies))


def VectorizeTag(tag):
    return np.array([1 if x in tag else 0 for x in Tag])


def CosineSimilarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


x = ratings[['user', 'movie']].values
y = ratings['user_rate'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# 生成训练集和测试集
n_user = len(ratings['userID'].unique())
n_item = len(ratings['movieID'].unique())

model = keras.models.load_model('model.keras')

#history = model.fit([x[:, 0], x[:, 1]], y, epochs=10, batch_size=256, validation_split=0.2)


def Recommendation(connection, user, tags, alpha=0.5):
    # print(user)
    # print(tags)
    # print(alpha)

    # 分割标签字符串
    custom_tag = tags.split(',')

    # 获取用户编码
    user_encoded = user2user_encoded.get(user)

    # 获取用户已看过的电影ID
    watched_movie_ids = ratings[ratings['userID'] == user]['movieID'].unique()

    # 计算标签向量和相似度
    movieInfo['tagVector'] = movieInfo['genre'].apply(VectorizeTag)
    custom_tag_vector = VectorizeTag(custom_tag)
    movieInfo['similarity'] = movieInfo['tagVector'].apply(lambda x: CosineSimilarity(x, custom_tag_vector))

    # 获取预测评分
    rating = model.predict([np.array([user_encoded] * len(MoviesEncoded)), np.array(MoviesEncoded)]).flatten()
    rating = np.clip(rating, 0, 1)  # 限制评分范围在 [0, 1]

    # 将预测评分映射到电影
    movie_rate = {movie_encoded2movie.get(MoviesEncoded[i]): rating[i] for i in range(len(Movies))}
    movieInfo['rate'] = movieInfo['movieID'].apply(lambda x: movie_rate.get(x))

    # 排除用户已看过的电影
    validMovieInfo = movieInfo[~movieInfo['movieID'].isin(watched_movie_ids)]

    # 获取评分与相似度加权的综合评分
    validMovieInfo['combined_score'] = alpha * validMovieInfo['rate'] + (1 - alpha) * validMovieInfo['similarity']

    # 从标签相似度排序的电影中，筛选出未看过且有效的电影，并计算综合评分
    valid_movies = validMovieInfo[
        (~validMovieInfo['movieID'].isin(watched_movie_ids)) &
        (validMovieInfo['movieID'].isin(valid_movie_ids['id'])) &
        (validMovieInfo['similarity'] > 0)
        ]

    # 计算综合评分，并按评分排序
    valid_movies = valid_movies.sort_values(by='combined_score', ascending=False)

    print(f"alpha:{alpha} tags:{tags} user_id:{user}")
    # 打印推荐电影表格
    print(f"{'电影名':<50}{'评分':<10}{'相似度':<10}{'综合评分':<10}{'类型':<30}")
    print("=" * 80)

    for index, movie_info in valid_movies.head(10).iterrows():
        print(f"{movie_info['name']:<50}{movie_info['rate']:<10.3f}{movie_info['similarity']:<10.3f}"
              f"{movie_info['combined_score']:<10.3f}{movie_info['genre']:<30}")

    # 获取前 10 部推荐电影
    top_movie_ids = valid_movies.head(10)['movieID'].tolist()

    # 从数据库中查询推荐电影的详细信息
    query = '''
        SELECT id, name, rate, img, genre, tag, country, summary
        FROM movies
        WHERE id IN ({seq})
    '''.format(seq=','.join(['?'] * len(top_movie_ids)))

    cursor = connection.cursor()
    cursor.execute(query, top_movie_ids)
    recommended_movies = cursor.fetchall()

    # 将查询结果转换为字典格式
    result = []
    for movie in recommended_movies:
        result.append({
            'id': movie[0],
            'name': movie[1],
            'rate': movie[2],
            'img': movie[3],
            'genre': movie[4],
            'tag': movie[5],
            'country': movie[6],
            'summary': movie[7]
        })

    return result

# 测试代码
connection = sqlite3.connect('movieInfo.db')
with connection:
    recommended_movies = Recommendation(connection, 25, '动作,冒险', 0.2)


connection.close()
