import json
import sqlite3
import pandas as pd
from collections import Counter

connection = sqlite3.connect('movieInfo.db')
cursor = connection.cursor()

# 从数据库获取电影数据，包括电影ID和对应的聚类ID
query = """
    SELECT id, name, tag, summary, cluster
    FROM movies
"""
movies_df = pd.read_sql(query, connection)
print(movies_df['tag'].head())

# 解析tag
movies_df['tag'] = movies_df['tag'].apply(eval)
print(movies_df['tag'].head())

# 为每个聚类创建单独的文件，保存该聚类的电影信息
for cluster_id in range(movies_df['cluster'].min(), movies_df['cluster'].max() + 1):
    cluster_movies = movies_df[movies_df['cluster'] == cluster_id]

    # 提取并统计标签
    cluster_tags = cluster_movies['tag']
    tag_counter = Counter([tag for tags in cluster_tags for tag in tags])

    # 获取频率最高的前5个标签，作为该类簇的主题
    common_tags = tag_counter.most_common(5)
    cluster_themes = [tag for tag, _ in common_tags]

    update_query = "UPDATE cluster_info SET summary = ? WHERE cluster_id = ?"
    cursor.execute(update_query, (json.dumps(cluster_themes), cluster_id))

    connection.commit()

# 将聚类概括文本存入数据库
# update_query = "UPDATE cluster_info SET summary = ? WHERE cluster_id = ?"
# cursor.execute(update_query, (json.dumps('''以犯罪为主题的经典电影，涵盖动作、黑帮、警匪、悬疑、黑色幽默等多种风格，涉及美国、英国、香港、法国、日本等多个国家的作品。这些电影通过紧张刺激的剧情和复杂多样的人物关系，揭示了犯罪背后的人性挣扎、道德冲突与社会问题。影片既有对暴力美学的刻画，也包含深刻的社会反思，展现了犯罪类型电影在艺术性和娱乐性上的高度融合，成为这一题材的重要代表作。
# '''), 19))

connection.commit()
# 关闭数据库连接
connection.close()
