import sqlite3
from flask import Flask, request as res, jsonify, request, json
from flask_cors import CORS

from PersonalizedRecommendations import Recommendation

Interactor = Flask(__name__)  # 创建Flask对象
CORS(Interactor)  # 允许跨域

@Interactor.route('/UserPortrait', methods=['GET'])
def UserPortrait():
    user_id = int(request.args.get('user_id'))
    connection = sqlite3.connect('movieInfo.db')
    cursor = connection.cursor()

    # 查询 user_portraits 表的数据
    query_portraits = '''
            SELECT cluster_id, movie_count, representative_movies
            FROM user_portraits
            WHERE user_id = ?
        '''
    cursor.execute(query_portraits, (user_id,))
    portrait_data = cursor.fetchall()

    # 查询 user_info 表的数据
    query_info = '''
            SELECT tags, genres, num
            FROM user_info
            WHERE user_id = ?
    '''
    cursor.execute(query_info, (user_id,))
    info_data = cursor.fetchall()

    # 查询 cluster_info 表中的 tags 和 summary
    query_cluster_info = '''
            SELECT tags, summary
            FROM cluster_info
            WHERE cluster_id = ?
    '''

    result_info = []
    for info in info_data:
        result_info.append({
            'tags': json.loads(info[0]),
            'genres': json.loads(info[1]),
            'num': info[2]
        })

    result_portrait = []
    for row in portrait_data:
        # 对每个 cluster_id 查询对应的 tags 和 summary
        cursor.execute(query_cluster_info, (row[0],))
        cluster_info = cursor.fetchone()
        tags = json.loads(cluster_info[0]) if cluster_info else []
        summary = cluster_info[1] if cluster_info else ""

        result_portrait.append({
            'cluster_id': row[0],
            'movie_count': row[1],
            'tags': tags,
            'summary': json.loads(summary),
            'representative_movies': json.loads(row[2])  # 解码JSON
        })

    connection.close()

    return jsonify({
        'user_info': result_info,
        'user_portraits': result_portrait
    })


@Interactor.route('/Recommendations', methods=['GET'])
def Recommendations():
    user_id = int(res.args.get('user_id'))
    tags = res.args.get('tags')
    personalizationLevel = float(res.args.get('personalizationLevel'))/100
    connection = sqlite3.connect('movieInfo.db')

    with connection:
        # 调用推荐函数获取电影推荐
        recommendations = Recommendation(connection, user_id, tags, personalizationLevel)

    connection.close()

    print(recommendations)

    return jsonify(recommendations)

@Interactor.route('/Filtering', methods=['GET'])
def Filtering():
    user_id = int(res.args.get('user_id'))
    connection = sqlite3.connect('movieInfo.db')
    cursor = connection.cursor()

    with connection:
        # 调用推荐函数获取电影推荐
        query = '''
            SELECT movie_id
            FROM user_recommendations
            WHERE user_id = ?
            LIMIT 16
        '''
        cursor.execute(query,(user_id,))
        movie_ids = cursor.fetchall()

        movie_ids = [row[0] for row in movie_ids]

        print(movie_ids)

        query_movies = '''
            SELECT id, name, rate, img, genre, tag, country, summary
            FROM movies
            WHERE id IN ({seq})
        '''.format(seq=','.join(['?'] * len(movie_ids)))

        cursor.execute(query_movies,movie_ids)
        recommendations = cursor.fetchall()

        print(recommendations)

    connection.close()

    return [
        {
            'id': movie[0],
            'name': movie[1],
            'rate': movie[2],
            'img': movie[3],
            'genre': movie[4],
            'tag': movie[5],
            'country': movie[6],
            'summary': movie[7]
        }
        for movie in recommendations
    ]

if __name__ == '__main__':
    Interactor.run(host='0.0.0.0', port=8927, debug=True)  # 运行Flask应用
