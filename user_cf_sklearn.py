# -*- coding: utf-8 -*-
"""
协同过滤算法示例（函数版，使用 sklearn）
==================================================
一、算法简介
--------------------------------------------------
协同过滤（Collaborative Filtering, CF）是推荐系统中最基础、最常见的方法之一。
它不依赖物品的内容特征，而是直接根据用户历史行为数据来做推荐。
本文件实现的是：
    基于用户的协同过滤（User-Based Collaborative Filtering）
其核心思想是：
    “如果两个用户过去喜欢的物品比较相似，
     那么其中一个用户喜欢、另一个用户还没有接触过的物品，
     就可以推荐给另一个用户。”
--------------------------------------------------
二、输入数据格式
--------------------------------------------------
输入是一个二维评分矩阵 ratings，格式为：
    ratings.shape = (用户数, 物品数)
例如：
    ratings = [
        [5, 3, 0, 1, 0],
        [4, 0, 0, 1, 2],
        [1, 1, 0, 5, 0],
        [0, 0, 5, 4, 0],
        [0, 1, 5, 4, 0],
    ]
说明：
    - 每一行表示一个用户
    - 每一列表示一个物品
    - 数值表示评分
    - 0 表示该用户没有给这个物品评分
--------------------------------------------------
三、算法流程
--------------------------------------------------
本实现的流程如下：

1. 计算每个用户的平均分
   因为不同用户的打分尺度不同，有的人偏高分，有的人偏低分，
   所以先计算每个用户自己的平均评分。

2. 对评分矩阵做中心化（mean-centering）
   对每个用户已评分项目执行：
       centered_rating = rating - user_mean

3. 使用 sklearn 计算用户之间的余弦相似度
   本文件使用：
       sklearn.metrics.pairwise.cosine_similarity

4. 预测评分
   对某个目标用户 u 和目标物品 i：
   - 找出对物品 i 打过分的其他用户
   - 选取与用户 u 最相似的前 k 个邻居
   - 使用加权平均估计评分

5. Top-N 推荐
   遍历该用户所有未评分物品，计算预测分，
   按分数从高到低排序，返回前 N 个。

--------------------------------------------------
四、预测公式
--------------------------------------------------
设：
    - u 为目标用户
    - i 为目标物品
    - v 为相似邻居用户

预测公式为：

                             Σ(sim(u,v) * (r(v,i) - mean(v)))
    pred(u,i) = mean(u) + -----------------------------------
                                      Σ|sim(u,v)|

其中：
    - sim(u,v) 表示用户 u 与用户 v 的相似度
    - r(v,i) 表示邻居 v 对物品 i 的评分
    - mean(u) / mean(v) 分别为用户平均分

--------------------------------------------------
五、依赖库
--------------------------------------------------
本文件依赖：
    1. numpy
    2. scikit-learn

安装方式：

    pip install numpy scikit-learn

--------------------------------------------------
六、优缺点
--------------------------------------------------
优点：
    1. 原理直观
    2. 实现简单
    3. 适合学习推荐系统基础
    4. 不依赖物品内容特征

缺点：
    1. 数据稀疏时效果会变差
    2. 新用户/新物品存在冷启动问题
    3. 用户很多时，相似度矩阵计算成本高
    4. 工业场景中通常会被更复杂模型替代

--------------------------------------------------
七、适用场景
--------------------------------------------------
适合：
    - 课程作业
    - 面试手写
    - 推荐系统入门
    - 小规模实验

不太适合：
    - 超大规模在线推荐系统
    - 高频实时推荐场景

--------------------------------------------------
八、如何运行
--------------------------------------------------
保存为 .py 文件后直接运行：

    python user_cf_sklearn.py
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def fit_user_cf(ratings):
    """
    训练基于用户的协同过滤模型。

    参数
    ----
    ratings : array-like, shape = (num_users, num_items)
        用户-物品评分矩阵，未评分位置使用 0 表示。

    返回
    ----
    model : dict
        训练好的模型，包含：
        - ratings: 原始评分矩阵
        - user_means: 每个用户的平均分
        - centered: 中心化后的评分矩阵
        - similarity: 用户相似度矩阵
        - num_users: 用户数
        - num_items: 物品数
    """
    ratings = np.array(ratings, dtype=float)

    if ratings.ndim != 2:
        raise ValueError("ratings 必须是二维矩阵")

    num_users, num_items = ratings.shape

    user_means = np.zeros(num_users, dtype=float)
    centered = ratings.copy()

    for u in range(num_users):
        rated_mask = ratings[u] > 0

        if np.any(rated_mask):
            user_means[u] = np.mean(ratings[u, rated_mask])
            centered[u, rated_mask] = ratings[u, rated_mask] - user_means[u]

        centered[u, ~rated_mask] = 0.0

    # 使用 sklearn 计算用户之间的余弦相似度
    similarity = cosine_similarity(centered)

    model = {
        "ratings": ratings,
        "user_means": user_means,
        "centered": centered,
        "similarity": similarity,
        "num_users": num_users,
        "num_items": num_items,
    }
    return model


def predict_score(model, user_id, item_id, k=5, min_rating=1.0, max_rating=5.0):
    """
    预测指定用户对指定物品的评分。

    参数
    ----
    model : dict
        fit_user_cf 返回的模型
    user_id : int
        用户索引
    item_id : int
        物品索引
    k : int, default=5
        参与预测的近邻数量
    min_rating : float, default=1.0
        评分下界
    max_rating : float, default=5.0
        评分上界

    返回
    ----
    float
        预测评分
    """
    ratings = model["ratings"]
    user_means = model["user_means"]
    similarity = model["similarity"]
    num_users = model["num_users"]
    num_items = model["num_items"]

    if not (0 <= user_id < num_users):
        raise IndexError("user_id 越界")
    if not (0 <= item_id < num_items):
        raise IndexError("item_id 越界")

    # 已有真实评分时，直接返回真实评分
    if ratings[user_id, item_id] > 0:
        return float(ratings[user_id, item_id])

    candidate_users = []
    for other_user in range(num_users):
        if other_user != user_id and ratings[other_user, item_id] > 0:
            sim = similarity[user_id, other_user]
            candidate_users.append((other_user, sim))

    # 如果没有候选邻居，回退到目标用户平均分
    if not candidate_users:
        return float(user_means[user_id])

    # 按相似度从高到低排序，取前 k 个
    candidate_users.sort(key=lambda x: x[1], reverse=True)
    neighbors = candidate_users[:k]

    numerator = 0.0
    denominator = 0.0

    for neighbor_id, sim in neighbors:
        # 负相似度一般不参与
        if sim <= 0:
            continue

        neighbor_rating = ratings[neighbor_id, item_id]
        neighbor_mean = user_means[neighbor_id]

        numerator += sim * (neighbor_rating - neighbor_mean)
        denominator += abs(sim)
    if denominator == 0:
        return float(user_means[user_id])
    pred = user_means[user_id] + numerator / denominator
    pred = max(min_rating, min(max_rating, pred))
    return float(pred)
def recommend_items(model, user_id, top_n=5, k=5):
    """
    为指定用户推荐 Top-N 未评分物品。

    参数
    ----
    model : dict
        fit_user_cf 返回的模型
    user_id : int
        用户索引
    top_n : int, default=5
        返回推荐数量
    k : int, default=5
        评分预测时使用的邻居数

    返回
    ----
    list[tuple]
        推荐结果列表，格式：
        [(item_id, predicted_score), ...]
    """
    ratings = model["ratings"]
    num_users = model["num_users"]
    num_items = model["num_items"]

    if not (0 <= user_id < num_users):
        raise IndexError("user_id 越界")
    results = []
    for item_id in range(num_items):
        if ratings[user_id, item_id] == 0:
            score = predict_score(model, user_id, item_id, k=k)
            results.append((item_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
def print_similarity_matrix(model):
    """
    打印用户相似度矩阵。
    """
    print("用户相似度矩阵：")
    print(np.round(model["similarity"], 4))
if __name__ == "__main__":
    ratings = [
        [5, 3, 0, 1, 0],
        [4, 0, 0, 1, 2],
        [1, 1, 0, 5, 0],
        [0, 0, 5, 4, 0],
        [0, 1, 5, 4, 0],
    ]
    model = fit_user_cf(ratings)
    print_similarity_matrix(model)
    print()
    user_id = 0
    item_id = 2
    pred = predict_score(model, user_id=user_id, item_id=item_id, k=2)
    print(f"用户 {user_id} 对物品 {item_id} 的预测评分：{pred:.4f}")
    print()

    recs = recommend_items(model, user_id=0, top_n=3, k=2)
    print(f"给用户 {user_id} 的推荐结果（Top-3）：")
    for item_id, score in recs:
        print(f"物品 {item_id}，预测评分 {score:.4f}")
