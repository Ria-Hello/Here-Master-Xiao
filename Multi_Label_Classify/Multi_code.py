import pandas as pd
import numpy as np
# 导入机器学习相关的工具包
from sklearn.model_selection import train_test_split, GridSearchCV  # 用于数据集划分和网格搜索
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.multioutput import MultiOutputClassifier  # 多标签分类器
from sklearn.metrics import accuracy_score, hamming_loss  # 用于评估模型性能的指标

# 加载评分数据
ratings = pd.read_csv(
    'data/ml-100k/u.data', 
    sep='\t', 
    header=None, 
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

# 加载电影数据
movies = pd.read_csv(
    'data/ml-100k/u.item', 
    sep='|', 
    header=None, 
    encoding='latin-1', 
    names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
           'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
)

# 加载用户数据
users = pd.read_csv(
    'data/ml-100k/u.user', 
    sep='|', 
    header=None, 
    names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
)

# 合并评分数据和电影数据
ratings_with_movies = pd.merge(ratings, movies, on='movie_id')

movie_data = ratings_with_movies[['movie_id', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]

movie_data = movie_data.drop_duplicates(subset = 'movie_id')

# 准备特征和标签
X = movie_data[['movie_id']]  # 特征X为电影ID
y = movie_data[['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]  # 标签y为18个电影类型

# 将数据集划分为训练集(80%)和测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建基础随机森林分类器
base_classifier = RandomForestClassifier()

# 由于需要预测多个标签(18个电影类型)，使用MultiOutputClassifier包装随机森林分类器
multi_classifier = MultiOutputClassifier(base_classifier)

# 定义需要调优的超参数网格
param_grid = {
    'estimator__n_estimators': [100, 200],  # 决策树的数量
    'estimator__max_depth': [10, 20],  # 树的最大深度
    'estimator__min_samples_split': [2, 5],  # 分裂内部节点所需的最小样本数
    'estimator__min_samples_leaf': [1, 2]  # 叶节点所需的最小样本数
}

# 使用网格搜索进行超参数调优，cv=5表示5折交叉验证
grid_search = GridSearchCV(
    multi_classifier,
    param_grid,
    cv=5,
    scoring='accuracy',  # 使用准确率作为评估指标
    n_jobs=-1  # 使用所有CPU核心并行计算
)

# 使用训练数据训练模型
grid_search.fit(X_train, y_train)

# 输出网格搜索找到的最佳参数组合
print("最佳参数:", grid_search.best_params_)

# 使用最佳模型对测试集进行预测
y_pred = grid_search.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
hamming = hamming_loss(y_test, y_pred)  # 计算Hamming损失(预测错误的标签比例)

print(f"准确率: {accuracy:.4f}")
print(f"Hamming损失: {hamming:.4f}")
