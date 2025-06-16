import sqlite3
import pandas as pd
import numpy as np


missing_value = ['none', '']
Source_Path = ''
conn = sqlite3.connect(Source_Path + 'movieInfo.db')
#comment = pd.read_csv(Source_Path + 'comment.csv')
movID = pd.read_csv(Source_Path + 'movID.csv')
movies = pd.read_csv(Source_Path + 'movies.csv')
person = pd.read_csv(Source_Path + 'person.csv')
ratInfo = pd.read_csv(Source_Path + 'ratInfo.csv')
relationships = pd.read_csv(Source_Path + 'relationships.csv')
usrID = pd.read_csv(Source_Path + 'usrID.csv')
usrRate = pd.read_csv(Source_Path + 'usrRate.csv', na_values=missing_value)
movieInfo = pd.read_sql_query('SELECT * FROM MovieInfo', conn)


# 将usrRate中的movieID转换为字符串
for (i, item) in enumerate(usrRate['movieID']):
    usrRate['movieID'][i] = str(item)

# 按年份分组
mov_group_by_year = movies.groupby('year').size().reset_index(name='counts')
mov_group_by_year = mov_group_by_year.sort_values(by='year', ascending=True)

# 计算每部电影的平均评分和评分人数
mov_mean_rate = pd.merge(usrRate, movieInfo, on=['movieID'])[['movieID', 'name', 'user_rate']].dropna().groupby(['name', 'movieID']).agg({'mean', 'count'}).reset_index()
mov_mean_rate.columns = ['name', 'movieID', 'count', 'mean']
mov_mean_rate = mov_mean_rate[mov_mean_rate['count'] >= 5]

ratings = pd.merge(usrRate, movieInfo, on=['movieID'])[['username','movieID', 'name', 'user_rate']].dropna()
ratings = pd.merge(ratings, usrID, on=['username'])[['userID','username', 'movieID', 'name', 'user_rate']].dropna()

# 查询数据库中的 id，并将其转换为字符串
valid_movie_ids = pd.read_sql_query("SELECT id FROM movies", conn)
valid_movie_ids['id'] = valid_movie_ids['id'].astype(str)

# 过滤时，将 movieID 和 id 进行字符串比较
ratings = ratings[ratings['movieID'].isin(valid_movie_ids['id'])]


print(len(ratings))

# 删除ratings中出现次数少于5的电影
ratings = ratings.groupby('movieID').filter(lambda x: len(x) >= 5)

user_ids = ratings["userID"].unique().tolist()
movie_ids = ratings["movieID"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
ratings["user"] = ratings["userID"].map(user2user_encoded)
ratings["movie"] = ratings["movieID"].map(movie2movie_encoded)

num_users = len(user_ids)
num_movies = len(movie_ids)
ratings["user_rate"] = ratings["user_rate"].values.astype(np.float32)
min_rating = min(ratings["user_rate"])
max_rating = max(ratings["user_rate"])