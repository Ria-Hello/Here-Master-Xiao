from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
import matplotlib.pyplot as plt
from torch import optim
import torch

def evaluate(model, test_loader):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁止计算梯度
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

train_data = datasets.MNIST('./DeepLearning/data/train',train = True,download = False,transform=transform)
test_data = datasets.MNIST('./DeepLearning/data/test',train=False,download = False,transform = transform)
train_data = Subset(train_data,range(20000))
test_data = Subset(test_data,range(1500))
#这个数据集太大了，我电脑算起来好慢，我裁剪一下确定能工作就行


#构建迭代器
train_loader = DataLoader(train_data,batch_size = 32,shuffle = True)
test_loader = DataLoader(test_data,batch_size = 32,shuffle = False)

image,label = next(iter(train_loader))

# 显示 MNIST 数据迭代器中的第一批数据
def visualize_images(images, labels):
    plt.figure(figsize=(5, 5))
    for i in range(32):
        plt.subplot(4,8, i + 1)
        plt.imshow(images[i].reshape([28,28]), cmap='gray')
        plt.title(f"{labels[i]}")
        plt.axis('off')
    plt.show()
# visualize_images(image,label)


#尝试自己设计一个模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,10))
    
    def forward(self,x):
        return self.model(x)
    
#定义一个损失函数
loss = nn.CrossEntropyLoss()

#设计一个优化器
my_MLP = MLP()
optimizer = optim.Adam(my_MLP.model.parameters(),lr = 0.003)

#训练开始：
epoch = 5
for i in range(epoch):
    total_loss = 0
    for j in range(1000//32):
        temp_loss = 0
        images,labels = next(iter(train_loader))
        #print('前向传播开始！')
        train_p = my_MLP.forward(images)
        #print('损失计算开始！')
        temp_loss = loss(train_p,labels)
        optimizer.zero_grad()
        #print('反向传播开始！')
        temp_loss.backward()
        optimizer.step()
        total_loss += temp_loss

    print(f'当前轮次{i+1},这个轮次的损失：{total_loss},准确率{evaluate(my_MLP,test_loader)}')




