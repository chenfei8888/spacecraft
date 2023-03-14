import torch
import numpy as np
import torch.nn as nn
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
from itertools import chain
from scipy.interpolate import make_interp_spline
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
# 直接多输出型

# 全局变量
state_num = 6
spacecraft_num = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## 定义网络，六*n个输出，xyz，x., y.,z.
# input_size: 6n
# output_size: 输出的步长，此处为1
# batch_size: 不重要
# n_outputs: 6n
# input_seq: 输入的序列，有很多数据
class LSTM(nn.Module):
    def  __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, n_outputs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fcs = [nn.Linear(self.hidden_size, self.output_size).to(device) for i in range(self.n_outputs)]
        for i in range(60):
            exec('self.fc{} = nn.Linear(self.hidden_size,self.output_size)'.format(i)) 
    def forward(self, input_seq):
        input_seq = input_seq.unsqueeze(0)
        # input_seq = m * 2, 2中是一个向量，2n*1
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = []
        x, y, z, xdot, ydot, zdot = self.fc1(output), self.fc2(output), self.fc3(output), self.fc4(output), self.fc5(output), self.fc6(output)
        x, y, z, xdot, ydot, zdot = x[:, -1, :], y[:, -1, :], z[:, -1, :], xdot[:, -1, :], ydot[:, -1, :], zdot[:, -1, :]
        for i in range(self.n_outputs):
            exec('output_{} = self.fc{}(output)'.format(i, i))
            exec('output_{} = output_{}[:, -1, :]'.format(i, i))
            exec('pred.append(output_{})'.format(i))
        preds = torch.stack(pred, dim = 0)
        return preds

# 这块就是这样，损失函数的定义
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = None
    def forward(self, y, y_pre):
        #output - output_predict
        # 反传
        res = torch.square_(y - y_pre)  
        res = torch.mean(res)
        self.loss = res
        return self.loss

# 机器学习的过程
class MLP(object):
    def __init__(self, path):
        # 数据路径
        self.path = path
    # 处理输入的数据为label的形式，一维
    def process(self, data, batch_size, shuffle):
        seq = []
        for i in range(int(data.shape[0] / (state_num * spacecraft_num))):
            for j in range(data.shape[1] - 2):  # 0-19
                train_seq = []
                train_label = []
                for p in range(state_num * spacecraft_num):
                    train_seq.append(data[i + p, 20]) # 把终点弄进去
                for p in range(state_num * spacecraft_num):
                    train_seq.append(data[i + p, j]) # 把起点_j弄进去
                for p in range(state_num * spacecraft_num):
                    train_label.append(data[i + p, j + 1])
                train_label = torch.FloatTensor(train_label)
                train_seq = torch.FloatTensor(train_seq)
                seq.append((train_seq, train_label))
        seq = DataLoader(dataset=seq, batch_size = batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
        return seq
    def train(self, learn_rate, args, Dtr, Val):
        # trainter迭代次数 perc训练集所占比例
        # iin 输入的索引， iout输出的索引 iin 为 21 和 i，输出为i+1
        # dis_pro展示比例, Dtr是训练集，Val是验证集，Dte是测试集
        # 加载数据，自己把训练集分出来
        
        #np.array(input)
        #input = np.array(input)
        # ydat 先不着急搞了
        # ydat = data[:ndat_train, iout]
        # print(input.shape)
        '''train_set, test_set = random_split(data, [ndat_train, ndat - ndat_train])
        train_loader = DataLoader(train_set, batch_size = 10, shuffle = True)
        test_loader = DataLoader(test_set, batch_size = 1)
        xdat_test = data[ndat_train:, iin]'''

        #数据归一化 
        '''scaler = StandardScaler()
        scaler.fit(data)
        datasc = scaler.transform(data)
        datasc = torch.from_numpy(datasc)'''

        #搭建模型
        input_size = state_num * spacecraft_num * 2
        hidden_size, num_layers = args.hidden_size, args.num_layers
        output_size = args.output_size
        n_outputs = state_num * spacecraft_num
        model = LSTM(input_size, hidden_size, num_layers, output_size, args.batch_size, n_outputs).to(device)
        model_loss = Loss().to(device)
        optimizer =  torch.optim.Adam(model.parameters(), lr = learn_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        #训练
        min_epochs = 10
        best_model = None
        min_val_loss = 5
        print(Dtr)
        for i in tqdm(range(min_epochs)):
            train_loss = []
            for (seq, label) in Dtr:
                seq = seq.to(device)
                label = label.to(device)
                total_loss = 0
                y_pred = model(seq)
                for k in range(n_outputs):
                    total_loss += model_loss(y_pred[k, :, :], label[:, k])
                total_loss /= n_outputs
                train_loss.append(model_loss.loss.item())
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            scheduler.step()   #调整学习率
            # 保存训练过程中在验证集上表现最好的模型
            val_loss = self.get_val_loss(args, model, Val)
            model.train()
            if i > min_epochs and val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
        state = {'models': best_model.state_dict()}
        torch.save(state)
    def get_val_loss(self, args, model, Val):
        val_loss = 0
        model_loss = Loss()
        for i in range(len(Val)):
            for j in range(len(Val)):
                for (seq, label) in Val:
                    y_pred = model(seq)
                    for k in range(args.output_size):
                        val_loss += model_loss(y_pred[k, :, :], label[:, k, :])
        return val_loss
    def test(self, args, Dte, path, m, n):
        # m, n 表示啥了
        pred = []
        y = []
        print('loading models...')
        input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
        output_size = args.output_size
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size)
        model.load_state_dict(torch.load(path)['models'])
        model.eval()
        print('predicting...')
        for (seq, target) in tqdm(Dte):
            target = list(chain.from_iterable(target.data.tolist()))
            y.extend(target)
            with torch.no_grad():
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                pred.extend(y_pred)
        y, pred = np.array(y), np.array(pred)
        y = (m - n) * y + n
        pred = (m - n) * pred + n
        print('mape:', self.get_mape(y, pred))
        # plot
        x = [i for i in range(1, 151)]
        x_smooth = np.linspace(np.min(x), np.max(x), 900)
        y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
        plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

        y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
        plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
        plt.grid(axis='y')
        plt.legend()
        plt.show()
    def get_mape(y, pred):
        y, pred = np.array(y), np.array(pred)
        return np.mean(np.abs((y - pred) / y)) * 100


class ARGS(object):
    def __init__(self, num_layers, input_size, output_size, perc, hidden_size, batch_size, step_size, gamma, n_outputs):
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.perc = perc
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.gamma = gamma
        self.n_outputs = n_outputs
def main():
    path = "/home/shibo/cxf_learn_trajectory/learning_trajectory/Data/spacecrafts_data.txt"
    args = ARGS(3, state_num * spacecraft_num, 1, 0.8, 20, 1, 1, 0.5, state_num * spacecraft_num)
    
    my_process = MLP(path)
    data = np.loadtxt(path)
    ndat = int(data.shape[0]/(state_num*spacecraft_num) * (data.shape[1] - 1))  #例子数,输入数据是十个航天器的起始点和终点
    train = data[ : int(data.shape[0] * 0.6)]
    val = data[int(data.shape[0] * 0.6) : int(data.shape[0] * 0.8)]
    test = data[int(data.shape[0] * 0.8) : ]
    print(type(train))
    train.tolist() 
    #m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])
    print(train.shape)
    shuffle = True
    Dtr = my_process.process(train, 1, shuffle)
    Val = my_process.process(val, args.batch_size, True)
    Dte = my_process.process(test, args.batch_size, False)
    learn_rate = 0.8
    my_process.train(learn_rate, args, Dtr, Val)
    path_module = "/home/shibo/cxf_learn_trajectory/learning_trajectory"
    #my_process.test(args, Dte, path_module, m, n)

if __name__ == '__main__':
    main()
