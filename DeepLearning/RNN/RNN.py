from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
vocab = {'我':0,'爱':1,'你':2,'丫':3,'的':4,'臭':5,'脚':6,'鸭':7,'子':8,'好':9,'吃':10}
idx_to_token = {idx : vocabs for vocabs,idx in vocab.items()}

class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, hidden_size, vocab, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = nn.RNN(len(vocab),hidden_size)
        self.vocab_size = len(vocab)
        self.num_hiddens = self.rnn.hidden_size
        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        
    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self,batch_size,device = 'cuda'):
        return  torch.zeros((1,batch_size, self.num_hiddens), device = device)
    
net = RNNModel(256,vocab)
net.to('cuda')
state = torch.zeros((1, 4, 256),device = 'cuda')
X = torch.tensor([[0,1,2,3],[2,1,0,3],[0,4,6,5],[2,4,5,6]])
                  #我爱你丫  你爱我丫   我的脚臭    你的臭脚
X = X.to('cuda')
Y, state_new = net(X, state)
print(Y.size())



def predict_ch8(prefix, num_preds, net, vocab, device):  
    """在prefix后面生成新字符"""
    state = net.begin_state(1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([idx_to_token[i] for i in outputs])

result = predict_ch8('我',2,net,vocab,'cuda')
print(result)

class TextDataset(Dataset):
    def __init__(self,text,vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.textdata = text
    
    def __len__(self):
        return len(self.textdata)
    
    def __getitem__(self, idx):
        text = self.textdata[idx]
        X = torch.tensor([self.vocab[char] for char in text[:-1]])  # 输入：去掉最后一个字符
        Y = torch.tensor([self.vocab[char] for char in text[1:]])  # 输出：去掉第一个字符
        return X, Y

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state = None
    for X, Y in train_iter:
        if use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if state is not None:
                state.detach_()  # 如果state不是None，执行detach操作
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        updater.zero_grad()
        l.backward()
        updater.step()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=True):
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr)
    predict = lambda prefix: predict_ch8(prefix, 2, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)

num_epochs, lr = 500, 1
text = ['我爱你丫','你的臭脚','你爱我丫','我的脚丫','你的臭脚','我的脚臭']
dataset = TextDataset(text,vocab)
dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True)
train_ch8(net, dataloader, vocab, lr, num_epochs, device = 'cuda')

result = predict_ch8('我',2,net,vocab,'cuda')
print(result)