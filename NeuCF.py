#from Utils import NeuCF
from DataPreprocessing import *
#import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

# 计算Precision@K和Recall@K
def precision_recall_at_k(predictions, ground_truth, k):
    # 获取Top-K推荐
    top_k_predictions = predictions[:k]

    # 计算推荐对的项目数量
    num_hit = len(set(top_k_predictions) & set(ground_truth))

    precision = num_hit / k if k > 0 else 0.0
    recall = num_hit / len(ground_truth) if len(ground_truth) > 0 else 0.0

    return precision, recall


# 计算nDCG@K
def ndcg_at_k(predictions, ground_truth_with_ratings, k):
    top_k_predictions = predictions[:k]

    # 将带有评分的ground_truth转换为字典，便于查找
    relevance_map = dict(ground_truth_with_ratings)

    # 计算DCG
    dcg = 0.0
    for i, pred_item in enumerate(top_k_predictions):
        if pred_item in relevance_map:
            # 获取真实评分作为相关性分数
            relevance = relevance_map[pred_item]
            dcg += relevance / np.log2(i + 2)  # i+2是因为i从0开始，log2(1)=0

    # 计算IDCG（理想情况下的DCG）
    # 理想排序是按真实评分降序排列
    ideal_order = sorted(relevance_map.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_order[:k]):
        idcg += rel / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


x = ratings[['user', 'movie']].values
y = ratings['user_rate'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# 生成训练集和测试集
n_user = len(ratings['userID'].unique())
n_item = len(ratings['movieID'].unique())

print("正在加载预训练的NeuCF模型...")
model = keras.models.load_model('model.keras')
print("模型加载完毕。")

print("\n--- 开始进行模型排序性能评估 ---")

# 评估参数
K = 10  # 评估Top-10的推荐
RATING_THRESHOLD = 4.0  # 定义用户“喜欢”的电影为评分大于等于4.0

# 划分训练集和测试集 (80%训练, 20%测试)
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

# 构建一个字典，存储每个用户在训练集中看过的电影
user_watched_in_train = train_df.groupby('userID')['movieID'].apply(list).to_dict()

# 构建测试集用户的 "Ground Truth"
# 即每个用户在测试集中评了高分的电影
test_user_ground_truth = {}
test_user_ground_truth_with_ratings = {}

# 按用户分组测试数据
test_user_groups = test_df.groupby('userID')

for user_id, group in test_user_groups:
    # 筛选出用户喜欢的电影
    relevant_movies = group[group['user_rate'] >= RATING_THRESHOLD]['movieID'].tolist()
    if relevant_movies:
        test_user_ground_truth[user_id] = relevant_movies
        # 存储带有真实评分的ground truth，用于nDCG计算
        test_user_ground_truth_with_ratings[user_id] = list(zip(group['movieID'], group['user_rate']))

print(f"数据集划分完毕。共 {len(test_user_ground_truth)} 位用户在测试集中有高分评价，将对他们进行评估。")

# 开始为测试用户生成推荐并评估
all_precisions = []
all_recalls = []
all_ndcgs = []

# 获取所有电影的编码，用于预测
all_movie_ids = ratings['movieID'].unique()
all_movie_encodeds = [movie2movie_encoded.get(m) for m in all_movie_ids]

# 使用tqdm显示进度条
for user_id in tqdm(test_user_ground_truth.keys(), desc="评估进度"):
    user_encoder = user2user_encoded.get(user_id)
    if user_encoder is None:
        continue

    # 预测用户对所有电影的评分
    user_array = np.full(len(all_movie_encodeds), user_encoder, dtype='int32')
    movie_array = np.array(all_movie_encodeds, dtype='int32')

    predictions = model.predict([user_array, movie_array], verbose=0).flatten()

    # 排除用户在训练集中已看过的电影
    watched_movies = user_watched_in_train.get(user_id, [])

    # 创建一个包含 (movieID, predicted_rating) 的元组列表
    movie_predictions = []
    for i, movie_id in enumerate(all_movie_ids):
        if movie_id not in watched_movies:
            movie_predictions.append((movie_id, predictions[i]))

    # 按预测评分降序排序，得到Top-K推荐列表
    movie_predictions.sort(key=lambda x: x[1], reverse=True)
    top_k_recs = [movie_id for movie_id, score in movie_predictions[:K]]

    # 计算指标
    ground_truth = test_user_ground_truth[user_id]
    ground_truth_ratings = test_user_ground_truth_with_ratings[user_id]

    precision, recall = precision_recall_at_k(top_k_recs, ground_truth, K)
    ndcg = ndcg_at_k(top_k_recs, ground_truth_ratings, K)

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_ndcgs.append(ndcg)

# 计算并打印平均指标
avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)
avg_ndcg = np.mean(all_ndcgs)

print("\n--- 模型排序性能评估结果 ---")
print(f"Precision@{K}: {avg_precision:.4f}")
print(f"Recall@{K}:    {avg_recall:.4f}")
print(f"nDCG@{K}:      {avg_ndcg:.4f}")
print("----------------------------")

# x = ratings[['user', 'movie']].values
# y = ratings['user_rate'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# # 生成训练集和测试集
# n_user = len(ratings['userID'].unique())
# n_item = len(ratings['movieID'].unique())
#
#
# model = keras.models.load_model('model.keras')

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

# 从测试集中准备评估数据
test_x = test_df[['user', 'movie']].values
test_y = test_df['user_rate'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# 在测试集上评估模型
loss, mean_squared_error = model.evaluate([test_x[:, 0], test_x[:, 1]], test_y, verbose=1)

print("\n--- 模型回归性能评估 (测试集) ---")
print(f"Test Loss: {loss:.4f}")
print(f"Test Mean Squared Error: {mean_squared_error:.4f}")

print("mean_squared_error: %.2f" % (mean_squared_error))

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