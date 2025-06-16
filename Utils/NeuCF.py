import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Dropout, Concatenate, BatchNormalization, Dot
from tensorflow.keras.optimizers import Adam


def NeuCF(num_users, num_movies, dim=50, l2=1e-6):
    l2_reg = tf.keras.regularizers.l2(l2)

    # 用户嵌入层
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = Embedding(input_dim=num_users, output_dim=dim, embeddings_regularizer=l2_reg)(user_input)
    user_embedding = Flatten()(user_embedding)
    user_embedding = BatchNormalization()(user_embedding)

    # 电影嵌入层
    movie_input = Input(shape=(1,), name='movie_input')
    movie_embedding = Embedding(input_dim=num_movies, output_dim=dim, embeddings_regularizer=l2_reg)(movie_input)
    movie_embedding = Flatten()(movie_embedding)
    movie_embedding = BatchNormalization()(movie_embedding)

    # GMF（Generalized Matrix Factorization）
    gmf = Dot(axes=1)([user_embedding, movie_embedding])

    # MLP 部分
    mlp = Concatenate()([user_embedding, movie_embedding])
    mlp = Dense(128, activation='relu')(mlp)
    mlp = Dropout(0.2)(mlp)
    mlp = Dense(64, activation='relu')(mlp)
    mlp = Dropout(0.2)(mlp)
    mlp = Dense(32, activation='relu')(mlp)

    # 拼接 GMF 和 MLP
    concat = Concatenate()([gmf, mlp])

    # 输出层
    output = Dense(1, activation='linear')(concat)
    # 构建模型
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model
