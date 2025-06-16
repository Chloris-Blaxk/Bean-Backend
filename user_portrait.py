import sqlite3
from collections import Counter

from flask import json


def get_user_portrait(user_id):

    print(f"Debug: 开始获取用户画像，user_id = {user_id}")

    connection = sqlite3.connect('movieInfo.db')
    cursor = connection.cursor()

    # 获取用户的观影记录及电影详细信息
    query_user_movies = '''
        SELECT M.id, M.name, M.cluster, M.rate, M.img, M.genre, M.tag, M.country, M.summary
        FROM movies M
        JOIN ratings U ON M.id = U.movieID
        WHERE U.userID = ?
    '''
    cursor.execute(query_user_movies, (user_id,))
    user_movies = cursor.fetchall()
    print(f"Debug: 查询到 {len(user_movies)} 条观影记录")

    # 按聚类统计信息
    cluster_stats = {}
    user_tags = [] # 用户tag，用于词云
    user_genres = [] # 用户观影类型，用于用户观影饼图
    num = len(user_movies) # 用户观影数量

    # 统计每部电影的信息
    for movie in user_movies:
        movie_id, movie_name, cluster_id, rate, img, genre, tags, country, summary = movie
        print(f"Debug: 处理电影: id={movie_id}, name={movie_name}, cluster={cluster_id}")

        if cluster_id not in cluster_stats:
            cluster_stats[cluster_id] = {
                "movie_count": 0,
                "tags": [],
                "representative_movies": []
            }

        # 更新聚类统计、用户tag和genre
        cluster_stats[cluster_id]["movie_count"] += 1
        cluster_stats[cluster_id]["tags"].extend(eval(tags))  # 解析标签
        user_tags.extend(eval(tags))
        user_genres.extend(genre.split('/'))

        # 每个类中展示三部代表电影
        if len(cluster_stats[cluster_id]["representative_movies"]) < 3:
            cluster_stats[cluster_id]["representative_movies"].append({
                "id": movie_id,
                "name": movie_name,
                "rate": rate,
                "img": img,
                "genre": genre,
                "tag": tags,
                "country": country,
                "summary": summary
            })

    # 用户的六十个频率最高的tag，用于生成词云
    user_60tags = {}
    for tag, number in Counter(user_tags).most_common(60):
        user_60tags[tag] = number

    # 用户的观影类型，用于饼图
    user_genre = {}
    for genre, number in Counter(user_genres).items():
        user_genre[genre] = number

    # 提取每个聚类的主题关键词
    for cluster_id, stats in cluster_stats.items():
        print(f"Debug: 处理聚类 cluster_id={cluster_id}")
        tag_counter = Counter(stats["tags"])
        cluster_stats[cluster_id]["keywords"] = [
            tag for tag, _ in tag_counter.most_common(5)
        ]

        # cluster_stats[cluster_id]["keywords60"] = [
        #     tag for tag, _ in tag_counter.most_common(60)
        # ]

        del cluster_stats[cluster_id]["tags"]  # 删除临时标签数据

    connection.close()
    print("Debug: 数据库连接已关闭")

    # 构建返回值
    response = []
    for cluster_id, stats in sorted(cluster_stats.items()):
        response.append({
            "cluster_id": cluster_id,
            "movie_count": stats["movie_count"],
            "keywords": stats["keywords"],
            # "keywords60": stats["keywords60"],
            "representative_movies": stats["representative_movies"]
        })

    print(f"Debug: 用户画像生成完成，共有 {len(response)} 个聚类")
    # print(user_60tags)
    # print(user_genre)
    # print(num)
#    print(stats["keywords"])
    return response, user_60tags, user_genre, num

# 预处理所有用户的用户画像信息
def store_all_user_portraits():
    connection = sqlite3.connect('movieInfo.db')
    cursor = connection.cursor()

    for user_id in range(1, 3328):  # 用户ID从1到3327
        portrait_data, user_60tags, user_genres ,num = get_user_portrait(user_id)

        # 插入用户画像数据
        query = '''
            INSERT INTO user_portraits (user_id, cluster_id, movie_count, keywords, representative_movies)
            VALUES (?, ?, ?, ?, ?)
        '''
        for cluster in portrait_data:
            cursor.execute(query, (
                user_id,
                cluster['cluster_id'],
                cluster['movie_count'],
                json.dumps(cluster['keywords']),  # 存储为JSON字符串
                # json.dumps(cluster['keywords60']),
                json.dumps(cluster['representative_movies'])
            ))

        connection.commit()

        query_user = '''
            INSERT INTO user_info (user_id, tags, genres, num)
            VALUES (?, ?, ?, ?)
        '''
        cursor.execute(query_user, (user_id, json.dumps(user_60tags), json.dumps(user_genres), num))

    connection.close()

#store_all_user_portraits()
#get_user_portrait(25)