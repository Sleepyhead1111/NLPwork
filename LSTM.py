import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8') # open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer

        if len(word) <= n_step:   # pad the sentence
            word = ["<pad>"]*(n_step+1-len(word)) + word

        for word_index in range(len(word)-n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index+n_step]]  # create (1~n-1) as input
            target = word2number_dict[word[word_index+n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch


def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  # open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))   # set to list

    word2number_dict = {w: i+2 for i, w in enumerate(word_list)}
    number2word_dict = {i+2: w for i, w in enumerate(word_list)}

    #add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"

    return word2number_dict, number2word_dict


#自实现单层LSTM
class LSTM_one(nn.Module):
    def __init__(self):
        super(LSTM_one, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        '''define the parameter of LSTM'''
        # 矩阵按位乘法(点乘),可以用torch.mul(a, b)实现，也可以直接用*实现
        # 遗忘门(forget)参数
        self.W_f = nn.Linear(n_hidden+emb_size, n_hidden, bias=True)
        # 输入门(input)参数
        self.W_i = nn.Linear(n_hidden+emb_size, n_hidden, bias=True)
        self.W_c = nn.Linear(n_hidden+emb_size, n_hidden, bias=True)
        # 输出门(output)参数
        self.W_o = nn.Linear(n_hidden+emb_size, n_hidden, bias=True)
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # 最终输出层
        self.W = nn.Linear(n_hidden, n_class, bias=True)

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]
        sample_size = X.size()[1]  # sample_size = batch_size
        '''do this LSTM forward'''
        c_0 = torch.zeros(sample_size, n_hidden)  # 记忆状态初值，设置为全0
        h_0 = torch.zeros(sample_size, n_hidden)  # 隐藏状态初值，设置为全0
        c = c_0    # 记忆状态初始化
        h = h_0    # 隐藏状态初始化
        for x in X:
            # 拼接形成[ h[t-1], x[t] ]
            catenate = torch.cat([h, x], dim=1)    #dim=1表示按行拼接,catenate: [batch_size, n_hidden+emb_size]
            # 遗忘门计算
            f = self.sigmoid(self.W_f(catenate))
            # 输入门计算
            i = self.sigmoid(self.W_i(catenate))
            cc = self.tanh(self.W_c(catenate))
            # 记忆更新
            c = torch.mul(f, c) + torch.mul(i, cc)
            # 输出门计算
            o = self.sigmoid(self.W_o(catenate))
            h = torch.mul(o, self.tanh(c))
        # model_output = nn.functional.softmax(self.W(h), dim=1)     # dim=1，对行作归一化
        model_output = self.W(h)
        return model_output, (h, c)    # 模型输出 = 预测概率, (隐藏状态, 记忆状态)


#自实现双层LSTM
class LSTM_double(nn.Module):
    def __init__(self):
        super(LSTM_double, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        '''define the parameter of LSTM'''
        # 矩阵按位乘法(点乘),可以用torch.mul(a, b)实现，也可以直接用*实现
        # 遗忘门(forget)参数
        self.W1_f = nn.Linear(n_hidden+emb_size, n_hidden, bias=True)
        self.W2_f = nn.Linear(n_hidden+n_hidden, n_hidden, bias=True)
        # 输入门(input)参数
        self.W1_i = nn.Linear(n_hidden+emb_size, n_hidden, bias=True)
        self.W1_c = nn.Linear(n_hidden+emb_size, n_hidden, bias=True)
        self.W2_i = nn.Linear(n_hidden+n_hidden, n_hidden, bias=True)
        self.W2_c = nn.Linear(n_hidden+n_hidden, n_hidden, bias=True)
        # 输出门(output)参数
        self.W1_o = nn.Linear(n_hidden+emb_size, n_hidden, bias=True)
        self.W2_o = nn.Linear(n_hidden+n_hidden, n_hidden, bias=True)
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # 最终输出层
        self.W = nn.Linear(n_hidden, n_class, bias=True)

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]
        sample_size = X.size()[1]  # sample_size = batch_size
        '''do this LSTM forward'''
        c1_0 = torch.zeros(sample_size, n_hidden)  # 第一层记忆状态初值为全0
        h1_0 = torch.zeros(sample_size, n_hidden)  # 第一层隐藏状态初值为全0
        c2_0 = torch.zeros(sample_size, n_hidden)     # 第二层记忆状态初值为全0
        h2_0 = torch.zeros(sample_size, n_hidden)     # 第二层隐藏状态初值为全0
        c1 = c1_0    # 第一层记忆状态初始化
        h1 = h1_0    # 第一层隐藏状态初始化
        c2 = c2_0    # 第二层记忆状态初始化
        h2 = h2_0    # 第二层隐藏状态初始化
        for x in X:
            # 第一层拼接形成[ h[t-1], x[t] ]
            catenate1 = torch.cat([h1, x], dim=1)    #dim=1表示按行拼接,catenate1: [batch_size, n_hidden+emb_size]
            # 第一层遗忘门计算
            f1 = self.sigmoid(self.W1_f(catenate1))
            # 第一层输入门计算
            i1 = self.sigmoid(self.W1_i(catenate1))
            cc1 = self.tanh(self.W1_c(catenate1))
            # 第一层记忆更新
            c1 = torch.mul(f1, c1) + torch.mul(i1, cc1)
            # 第一层输出门计算
            o1 = self.sigmoid(self.W1_o(catenate1))
            h1 = torch.mul(o1, self.tanh(c1))
            # 第二层拼接
            catenate2 = torch.cat([h2, h1], dim=1)  # dim=1表示按行拼接,catenate: [batch_size, n_hidden+emb_size]
            # 第二层遗忘门计算
            f2 = self.sigmoid(self.W2_f(catenate2))
            # 第二层输入门计算
            i2 = self.sigmoid(self.W2_i(catenate2))
            cc2 = self.tanh(self.W2_c(catenate2))
            # 第二层记忆更新
            c2 = torch.mul(f2, c2) + torch.mul(i2, cc2)
            # 第二层输出门计算
            o2 = self.sigmoid(self.W2_o(catenate2))
            h2 = torch.mul(o2, self.tanh(c2))
        # model_output = nn.functional.softmax(self.W(h2), dim=1)     # dim=1，对行作归一化
        model_output = self.W(h2)
        h = [h1, h2]    # 隐藏状态 = [第一层隐藏状态, 第二层隐藏状态]
        c = [c1, c2]    # 记忆状态 = [第一层记忆状态, 第二层记忆状态]
        return model_output, (h, c)    # 模型输出 = 预测概率, (隐藏状态, 记忆状态)


def train_LSTM():
    model = LSTM_double()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output[0], target_batch)
            ppl = math.exp(loss.item())
            # if (count_batch + 1) % 50 == 0:
            #     print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
            #           'lost =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(word2number_dict, n_step)
        all_valid_batch.to(device)
        all_valid_target.to(device)
        
        total_valid = len(all_valid_target)*128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output[0], valid_target)
                total_loss += valid_loss.item()
                count_loss += 1
          
            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'lost =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch+1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/lstm_model_epoch{epoch+1}.ckpt')

def test_LSTM(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  #load the selected model

    #load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(word2number_dict, n_step)
    total_test = len(all_test_target)*128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('lost =','{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

if __name__ == '__main__':
    torch.manual_seed(3407)

    n_step = 5 # number of cells(= number of Step)
    n_hidden = 5 # number of hidden units in one cell
    batch_size = 512 #batch size
    learn_rate = 0.001
    all_epoch = 200 #the all epoch for training
    emb_size = 128 #embeding size
    save_checkpoint_epoch = 100 # save a checkpoint per save_checkpoint_epoch epochs
    train_path = 'data/train.txt' # the path of train dataset

    word2number_dict, number2word_dict = make_dict(train_path) #use the make_dict function to make the dict
    print("The size of the dictionary is:", len(word2number_dict))

    n_class = len(word2number_dict)  #n_class (= dict size)

    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    print("The number of the train batch is:", len(all_input_batch))

    all_input_batch = torch.LongTensor(all_input_batch).to(device)   #list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)

    print("\nTrain the LSTM……………………")
    train_LSTM()

    # print("\nTest the LSTM……………………")
    # select_model_path = "models/LSTM_model_epoch2.ckpt"
    # test_LSTM(select_model_path)