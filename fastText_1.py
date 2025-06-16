import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from gensim.models import fasttext
from tqdm import tqdm
from collections import Counter

connection = sqlite3.connect('movieInfo.db')
cursor = connection.cursor()

# 加载FastText预训练模型
print("Loading FastText model...")
fasttext_model = fasttext.load_facebook_vectors('cc.zh.300.bin')  # 300维中文词向量

# 读取电影数据
movies_df = pd.read_csv('movies.csv')

# 定义一个获取标签平均词向量的函数
def get_tag_vector_average(tags):
    vectors = []
    for tag in tags:
        if tag in fasttext_model.key_to_index:  # 检查词是否存在
            vectors.append(fasttext_model[tag])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(fasttext_model.vector_size)  # 若标签没有词向量则返回零向量

# 提取标签并计算每部电影的特征向量
movies_df['tags'] = movies_df['tags'].apply(eval)  # 将字符串形式的列表转换为实际的列表
movies_df['tag_vector'] = [get_tag_vector_average(tags) for tags in tqdm(movies_df['tags'], desc="Processing Tags")]

# 构造特征矩阵
feature_matrix = np.stack(movies_df['tag_vector'].values)

# 使用PCA降维以加速聚类
pca = PCA(n_components=50, random_state=42)  # 将维度降到50
feature_matrix_pca = pca.fit_transform(feature_matrix)

# 使用KMeans进行聚类
num_clusters = 20

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
movies_df['cluster'] = kmeans.fit_predict(feature_matrix_pca)

#将聚类结果写入数据库
# for index, row in movies_df.iterrows():
#     update_query = "UPDATE movies SET cluster = ? WHERE id = ?"
#     cursor.execute(update_query, (int(row['cluster']), row['id']))
#
# connection.commit()
# connection.close()
# print("Database updated successfully.")

# 计算WCSS和轮廓系数
wcss = kmeans.inertia_
silhouette_avg = silhouette_score(feature_matrix_pca, movies_df['cluster'])

# 输出聚类结果和评价指标
print(f"WCSS (Sum of Squared Distances): {wcss}")
print(f"Silhouette Score: {silhouette_avg}")

# 使用t-SNE降维进行可视化
tsne = TSNE(n_components=2, random_state=42)
feature_matrix_2d = tsne.fit_transform(feature_matrix_pca)

# 绘制聚类结果
plt.figure(figsize=(12, 8))
scatter = plt.scatter(feature_matrix_2d[:, 0], feature_matrix_2d[:, 1], c=movies_df['cluster'], cmap='viridis',
                      alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title("t-SNE Visualization of Movie Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()

# 输出每个聚类的部分结果和主题标签
print("\nClustered Movies (每个类20部代表电影):")
cluster_themes = {}

for cluster_id in range(num_clusters):
    print(f"\nCluster {cluster_id}:")
    cluster_movies = movies_df[movies_df['cluster'] == cluster_id]

    # 提取并统计标签
    cluster_tags = cluster_movies['tags']
    tag_counter = Counter([tag for tags in cluster_tags for tag in tags])

    # 获取频率最高的前5个标签，作为该类簇的主题
    common_tags = tag_counter.most_common(5)
    cluster_themes[cluster_id] = [tag for tag, _ in common_tags]

    # 保存类簇信息到文件，以便后续处理
    output_data = []
    output_data.append(f"Cluster {cluster_id} - 主题标签: {', '.join(cluster_themes)}\n")
    output_data.append("电影列表:\n")

    # 输出该聚类的电影名、标签和简介
    for _, row in cluster_movies.iterrows():
        movie_info = f"电影名: {row['name']}\n简介: {row['summary']}\n标签: {', '.join(row['tags'])}\n"
        output_data.append(movie_info)

    # 保存数据到文件，文件名为 'cluster_{cluster_id}.txt'
    output_filename = f"cluster_{cluster_id}.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.writelines(output_data)

    print(f"Cluster {cluster_id} information saved to {output_filename}")