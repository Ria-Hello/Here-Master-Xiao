import torch 
from torch import nn
from torch.utils.data import Dataset,DataLoader
Chinese_text = ['我爱你',
                '你好',
                '早上好',
                '你下午去哪里了',
                '早上我要去打篮球',
                '请讲普通话',
                '明天上午我要去看医生',
                '今天的晚饭有点不新鲜啊',
                '你也这样觉得吗',
                '其实我更喜欢吃清淡点的菜',
                '麻辣香锅确实挺好吃的',
                '我总觉得哪里怪怪的',
                '也许我应该去医院看看',
                '请不要这么说话',
                '你喜欢看电影吗'
                ]
English_text = ['I love you',
                'Hi',
                'Good morning',
                'Where did you go in the afternoon',
                'I am going to play basketball in the morning',
                'Please speak Mandarin',
                'I am going to see a doctor tomorrow morning',
                'Today\'s dinner is a bit unfresh',
                'Do you feel the same way',
                'Actually, I prefer to eat lighter dishes',
                'The spicy fragrant pot is really delicious',
                'I always felt weird',
                'Maybe I should go to the hospital and have a look',
                'Please don\'t talk like that',
                'Do you like watching movies',
                ]
def text_process(Chinese_text,English_text):
    #文本处理函数，传入两个未加工的文本列表，
    #            传出一个总文本total_text（三维张量,total_text[0]和total_text[1]分别是俩个文本列表）,可供观察使用
    #            传出一个总词表total_vocab
    #            以及文本中最长的句子的词元数量
    i = 4
    temp_total_text = []
    total_text = []
    total_vocab = {'<pad>':0,'<unk>':1,'<bos>':2,'<eos>':3}
    max_length = 0

    for line in Chinese_text:
    #处理中文文本
        temp = ['<bos>']
        #获取每行文本时先给每个文本开头加上<bos>
        if len(line) > max_length:
            max_length = len(line)
        #提取文本中的每个字顺便统计句子长度（方便后续设置文本截断长度）
        for word in line:
            if word not in total_vocab:
                total_vocab[word] = i
                i += 1
            #将文本中出现的汉字添加进字典中
            temp += f'{word}'
        temp += ['<eos>']
        #句子末端加上<eos>
        temp_total_text += [temp]
    total_text += [temp_total_text]


    temp_total_text = []
    #处理英文文本
    for line in English_text:
        list = line.split(' ')
        #将英文文本按词划分
        if len(list) > max_length:
            max_length = len(list)
        #统计最长英文句子单词次数
        temp = ['<bos>'] + list + ['<eos>']
        #加上句子的开头和结尾标志
        temp_total_text += [temp]
        for word in list:
            if word not in total_vocab:
                total_vocab[word] = i
                i += 1
        #将英文词添加进字典
    total_text += [temp_total_text]

    index_total_text = []
    #将句子转换为索引表示
    for i in range(2):
        temp_index = []
        for line in total_text[i]:
            index = []
            for token in line:
                index += [total_vocab[token]]
            temp_index += [index]
        index_total_text += [temp_index]
    return total_text,index_total_text,total_vocab,max_length
total_text,index_total_text,total_vocab,max_length = text_process(Chinese_text=Chinese_text,English_text=English_text)

#print(len(total_vocab))
#词表长度是130，总共有130个token

#将每个句子padding成max_length长度
def pad_text(index_total_text,padding_length,padding_index):
    for i in range(len(index_total_text)):
        for j in range(len(index_total_text[i])):
            line = index_total_text[i][j]
            if len(line)<(padding_length + 2):
                index_total_text[i][j] = line + [padding_index]*((padding_length + 2) - len(line))
            else:
                index_total_text[i][j] = line[:padding_length+2]
    return index_total_text
#【！】【！】【！】【！】这里粗略地使用了两个全循环，理论上这是一个空间复杂度为O(n^2)的算法，可以接受吗？

index_total_text = pad_text(index_total_text,max_length,total_vocab['<pad>'])

# for i in range(len(index_total_text)):
#     for j in range(len(index_total_text[0])):
#         print(len(index_total_text[i][j]))
#查看padding是否正常
embedding_size = 768
emb = nn.Embedding(len(total_vocab),embedding_size)
# for i in total_vocab:
#     total_vocab[i] = emb.weight[total_vocab[i]]
#通过Embedding，vocab中的键值转换为768维的向量

class text_dataset(Dataset):
    def __init__(self,index_total_text,total_text):
        self.total_text = total_text
        self.index_total_text = index_total_text

    def __len__(self):
        return len(self.total_text[0])

    def __getitem__(self,idx):
        index_total_text = self.index_total_text
        total_text = self.total_text
        return torch.tensor(index_total_text[0][idx]),torch.tensor(index_total_text[1][idx]),len(total_text[0][idx]),len(total_text[1][idx])
    
textDataset = text_dataset(index_total_text,total_text)
dataloader = DataLoader(dataset=textDataset,batch_size=4,shuffle=True)

# for batch_idx, (Cidx,Eidx, len1, len2) in enumerate(dataloader):
#     print(f"Batch {batch_idx + 1}:")
#     print("Eidx_data:\n", Cidx)
#     print("Cidex_data:\n", Eidx)
#     print("len1:\n", len1)
#     print("len2:\n", len2)
#     print("------" * 10)
#查看迭代器内部数据是否符合预期

class Encoder(nn.Module):
    def __init__(self,Embedding,hidden_size,layers):
        super(Encoder, self).__init__()
        self.Embedding = Embedding
        self.rnn = nn.RNN(embedding_size,hidden_size=hidden_size,num_layers=layers)
    
    def forward(self,x):
        x = self.Embedding(x).permute(1,0,2)
        output,state = self.rnn(x)
        return output,state
# for batch_idx, (Cidx,Eidx, len1, len2) in enumerate(dataloader):
#     output,state = encoder(Cidx)
#     print(output.shape)
#     print(state.shape)
#验证encoder工作正常（给一个中文序列，输出隐藏状态）
class Decoder(nn.Module):
    def __init__(self,Embedding,hidden_size,layer):
        super(Decoder,self).__init__()
        self.Embedding = Embedding
        self.rnn = nn.RNN(embedding_size,hidden_size=hidden_size,num_layers=layer)
        self.linear = nn.Linear(hidden_size,embedding_size)

    def forward(self,x,state):
        En_state = state
        x = self.Embedding(x).permute(1,0,2)
        output,De_state = self.rnn(x,En_state)
        return self.linear(output),De_state
    

def run_one_iter(encoder,decoder,dataloader):
        for C_text,E_text,_,E_length in dataloader:
            _,state = encoder(C_text)
            output,de_state = decoder(E_text,state)
            print('*'*60)
            print("这里是待翻译的句子：")
            show(C_text)
            print('*'*60)
            return output.permute(1,0,2)
encoder = Encoder(emb,1024,1)
decoder = Decoder(emb,1024,1)

def similarity(emb,encoder,decoder,dataloader):

    output = run_one_iter(encoder=encoder,decoder=decoder,dataloader=dataloader)
    # print(output.shape)
    dot_output = torch.matmul(output,emb.T)
    # print(dot_output.shape)
    predicted_indices = torch.argmax(dot_output, dim=-1)  
    #打印预测的句子
    print('*'*60)
    print("这是翻译结果：")
    show(predicted_indices)
    print('*'*60)

def show(list):
    idx_to_word = {v: k for k, v in total_vocab.items()}
    predicted_tokens = []
    for i in range(len(list)):
        sentence_tokens = [idx_to_word[idx.item()] for idx in list[i] ]
        predicted_tokens.append(sentence_tokens)
    for i in range(len(predicted_tokens)):
        for j in range(len(predicted_tokens[i])):
            if predicted_tokens[i][j] == '<eos>':
                predicted_tokens[i] = predicted_tokens[i][:j+1]
                break
    for i in range(len(predicted_tokens)):
        print(f"{''.join(' '+ predicted_tokens[i][j] for j in range(len(predicted_tokens[i])))}")
            
similarity(emb.weight,encoder,decoder,dataloader)


def train(encoder,decoder,epoch,en_optimizer,de_optimizer,dataloader):
    encoder = encoder
    decoder = decoder
    encoder_optimizer = en_optimizer
    decoder_optimizer = de_optimizer
    for i in range(epoch):
        total_loss = 0
        for C_text,E_text,C_Len,E_len in dataloader:
            mask = torch.zeros((E_text.size(0), E_text.size(1)), dtype=torch.float32)
            _,state= encoder(C_text)
            output,_ = decoder(E_text,state)
            output = output.permute(1,0,2)
            index_output = torch.matmul(output,emb.weight.T)
            for j, length in enumerate(E_len):
                mask[j, :length] = 1
            mask = mask.reshape(-1)
            index_output = index_output.reshape(-1,index_output.size(-1))
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(index_output,E_text.reshape(-1))
            if mask.sum() > 0:
                loss = (loss * mask).sum() / mask.sum()
            else:
                continue
            total_loss += loss.item()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        print(f'这是第{i+1}轮训练：总损失为{total_loss}')
train(encoder,decoder,5,torch.optim.Adam(encoder.parameters(),0.0003),torch.optim.Adam(decoder.parameters(),0.01),dataloader)


similarity(emb.weight,encoder,decoder,dataloader)

def translate(sentence,emb,encoder,decoder):
    sentence_index = [2]
    for char in sentence:
        if char in total_vocab:
            sentence_index.append(total_vocab[char])
        else:
            print('这个句子出现了没见过的字，不翻译')
            return 0
    sentence_index.append(3)
    _,state = encoder(torch.tensor(sentence_index).unsqueeze(0))
    predicted_indices = []  # 存储解码出的索引
    decoder_input = torch.tensor([[2]])
    for _ in range(20):  # 限制最大解码长度
        output, state = decoder(decoder_input, state)  # 解码器输出
        index_output = torch.matmul(output.squeeze(0), emb.weight.T)  # 转换为词表分布
        predicted_index = torch.argmax(index_output, dim=-1).item()  # 获取最高概率的词索引

        if predicted_index == 3:  # 遇到结束标记 <eos>，停止解码
            break

        predicted_indices.append(predicted_index)
        decoder_input = torch.tensor([[predicted_index]])  # 更新解码器输入

    # 将索引转换为单词并打印翻译结果
    idx_to_word = {v: k for k, v in total_vocab.items()}
    translated_sentence = ''.join(idx_to_word[idx] for idx in predicted_indices)
    print(f"翻译结果: {translated_sentence}")
    

translate('你爱我',emb,encoder,decoder)





