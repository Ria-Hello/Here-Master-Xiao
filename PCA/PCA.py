import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,5))

data = np.random.randn(400,3)
data[:,0] += 6
data[:,1] += 2

print('原始数据形状：',data.shape)
#随机生成400个点，每个点有五个维度特征，将其中第一维和第二维数据偏移一下，试着分开，然后打印

#**********************画图函数不用在意***********************************************
ax1 = fig.add_subplot(121, projection='3d')  # 1行2列第1个，3D图
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o', s=30, alpha=0.5)
ax1.set_title('Raw_data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
#************************************************************************************

#👇去中心化
centered_data = data - np.mean(data,axis = 0)

#👇计算协方差矩阵
cov_matrix = np.cov(centered_data,rowvar=False)
print(cov_matrix)

#👇从协方差矩阵中提取特征向量以及特征值
eigvalue,eigvector = np.linalg.eig(cov_matrix)
print(eigvalue)
print(eigvector)

#👇利用特征值进行排序，找出其中变化较大的前k个特征向量
k = 2
sorted_indices = np.argsort(eigvalue)[::-1]
sorted_eigvalue = eigvalue[sorted_indices]
sorted_eigvector = eigvector[:,sorted_indices]

selected_vector = sorted_eigvector[:, :k]

#👇降维数据
reduce_data = np.dot(centered_data,selected_vector)

#**********************画图函数不用在意***********************************************
ax2 = fig.add_subplot(122)  # 1行2列第2个
ax2.scatter(reduce_data[:, 0], reduce_data[:, 1], c='r', marker='o', s=30, alpha=0.5)
ax2.set_title('After_PCA')
ax2.set_xlabel('main_component_1')
ax2.set_ylabel('main_component_2')
#************************************************************************************

plt.tight_layout()
plt.show()