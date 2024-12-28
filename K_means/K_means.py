from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

MAX_ITER = 100
samples_number = 500
clusters_number = 5
features_number = 2
clusters_std = 1.5

X,y = make_blobs(n_samples = samples_number,n_features = features_number,centers = clusters_number,cluster_std = clusters_std,random_state=42)
#生成用于聚类的点。
# plt.scatter(X[:,0],X[:,1],c = y)
# plt.xlabel('feature_1')
# plt.ylabel('feature_2')
# plt.title('Raw_dataset')
# plt.show()
def O_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
#计算欧氏距离，就是平方差
def K_means(samples_list ,k, max_iter = MAX_ITER):
    num_samples,num_features = samples_list.shape
    centers = samples_list[np.random.choice(num_samples,k,replace=False)]
    #随机生成K个点作为簇的中心
    for _ in range(max_iter):
        labels = np.zeros(num_samples)
        for i in range(num_samples):
            distances = [O_distance(samples_list[i],center) for center in centers]
            labels[i] = np.argmin(distances)
        #这里计算所有点到随机生成的点之间的距离，并根据距离给每一个点打上标签

        new_centers = np.zeros([k,num_features])
        for i in range(k):
            clusters_points = samples_list[labels == i]
            #取出各个标签下的点
            if len(clusters_points) > 0:
                new_centers[i] = np.mean(clusters_points, axis = 0)
                #新的点为同一标签下点的均值
        
        if np.all(centers == new_centers):
            break

        centers = new_centers
    return centers,labels

centers,labels = K_means(X,clusters_number)

plt.scatter(X[:,0],X[:,1],c = labels)
plt.scatter(centers[:,0],centers[:,1],marker= 'x' ,c = 'red')
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.show()




        