#导包
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils import data
from torch import nn

#自定义一个造数据函数，
# 传入：真实的权重、偏置、需要的样本数
# 输出：构建的所有样本的特征、对应的标签
def synthetic_data(w, b, num_examples):  
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

#设置真实的权重和偏置
true_w = torch.tensor([-1.54,2.89])
true_b = 89.76

#借助synthetic_data造数据：
data_X,data_Y = synthetic_data(true_w,true_b,500)

# # 查看样本在三维空间中的分布
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 确保 data_X 和 data_Y 是一维张量
# ax.scatter(data_X[:, 0].detach().numpy(), 
#            data_X[:, 1].detach().numpy(), 
#            data_Y.squeeze().detach().numpy(), 
#            alpha=1, s=100)

# ax.set_xlabel('X[0]')
# ax.set_ylabel('X[1]')
# ax.set_zlabel('Y')
# plt.title('Manual Data')
# plt.show()


#造迭代器，根据批次大小给数据
def load_array(data_arrays, batch_size, is_train=True):  
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((data_X, data_Y), batch_size)

#利用Sequential函数构建模型
net = nn.Sequential(nn.Linear(2, 1))
#用net变量改变初始值
net[0].weight.data.normal_(0, 0.01) #改变第1层模型的权重初始值变成均值为0方差为0.01的分布
net[0].bias.data.fill_(0)           #改变第一层模型的偏置初始值

#构造损失函数
loss = nn.MSELoss()

#构造优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(data_X), data_Y)
    print(f'epoch {epoch + 1}, loss {l:f}')
    print('weight:[{},{}]\n bias:{}\n'.format(net[0].weight[0][0].item(),net[0].weight[0][1].item(),net[0].bias.item()))

    #整个设计思路按照：获取数据、构建迭代器、构建模型、损失函数设计、优化器设计 5个步骤来实现，有高级API用起来还是很方便的。
    #softmax大差不差的流程，我挑个小项目做一下吧（涉及MLP）。
