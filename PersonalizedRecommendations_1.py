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
    print(user)
    print(tags)
    print(alpha)

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

    # 从数据库中获取推荐的电影ID
    query = '''
        SELECT movie_id
        FROM user_recommendations
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 25
    '''
    cursor = connection.cursor()
    cursor.execute(query, (user,))
    rate_top_movie_ids = [row[0] for row in cursor.fetchall()]
    #print("Database fetched rate_top_movie_ids:", rate_top_movie_ids)

    # 查询数据库中的 id，并将其转换为字符串
    valid_movie_ids = pd.read_sql_query("SELECT id FROM movies", connection)
    valid_movie_ids['id'] = valid_movie_ids['id'].astype(str)

    # 获取标签相似度排序的电影ID（未看过且有效）
    sim_top_movie_ids = movieInfo[
        (~movieInfo['movieID'].isin(watched_movie_ids)) &
        (movieInfo['movieID'].isin(valid_movie_ids['id'])) &
        (movieInfo['similarity'] > 0)
        ].sort_values('similarity', ascending=False).head(25)['movieID'].tolist()
    #print("Filtered sim_top_movie_ids:", sim_top_movie_ids)
    #print(sim_top_movie_ids)

    # 根据 alpha 参数随机采样
    alpha = math.floor(alpha * 10 + 0.5)
    #print(alpha)
    
    if not rate_top_movie_ids or not sim_top_movie_ids:
        print("No valid movies to recommend")
        return []


    top_movie_ids = random.sample(rate_top_movie_ids, min(alpha, len(rate_top_movie_ids))) + \
                    random.sample(sim_top_movie_ids, min(10 - alpha, len(sim_top_movie_ids)))
    #print(top_movie_ids)

    # 从数据库中查询推荐电影的详细信息
    query = '''
        SELECT id, name, rate, img, genre, tag, country, summary
        FROM movies
        WHERE id IN ({seq})
    '''.format(seq=','.join(['?'] * len(top_movie_ids)))

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
        #print(movie[1])

    return result


# connection = sqlite3.connect('movieInfo.db')
# with connection:
#     # 调用推荐函数获取电影推荐
#     print(Recommendation(connection,25, '悬疑,惊悚',1))

# connection.close()