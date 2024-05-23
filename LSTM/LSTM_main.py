# -*- coding = utf-8 -*-
# @Time : 2024/5/22 9:39
# @Author : ChiXiaoWai
# @File : LSTM_main.py
# @Project : time_series
import argparse
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib
from torch.nn.utils import weight_norm
matplotlib.use('TkAgg')
plt.switch_backend('agg')
# 随机数种子
np.random.seed(0)


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def plot_loss_data(data):
    # 使用Matplotlib绘制线图
    plt.figure()

    plt.plot(data, marker='o')

    # 添加标题
    plt.title("loss results Plot")

    # 显示图例
    plt.legend(["Loss"])

    plt.show()


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, config):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == 'MS':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_csv(config.data_path)  # 填你自己的数据地址,自动选取你最后一列数据为特征列 # 添加你想要预测的特征列
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口

    # 将特征列移到末尾
    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = df.columns[1:]
    df_data = df[cols_data]

    # 这里加一些数据的预处理, 最后需要的格式是pd.series
    true_data = df_data.values

    # 定义标准化优化器
    # 定义标准化优化器
    scaler = StandardScaler()
    scaler.fit(true_data)

    train_data = true_data[int(0.3 * len(true_data)):]
    valid_data = true_data[int(0.2 * len(true_data)):int(0.30 * len(true_data))]
    test_data = true_data[:int(0.2 * len(true_data))]
    print("训练集尺寸:", len(train_data), "测试集尺寸:", len(test_data), "验证集尺寸:", len(valid_data))

    # 进行标准化处理
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    valid_data_normalized = scaler.transform(valid_data)

    # 转化为深度学习模型需要的类型Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)

    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, config)

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print("通过滑动窗口共有训练集数据：", len(train_inout_seq), "转化为批次数据:", len(train_loader))
    print("通过滑动窗口共有测试集数据：", len(test_inout_seq), "转化为批次数据:", len(test_loader))
    print("通过滑动窗口共有验证集数据：", len(valid_inout_seq), "转化为批次数据:", len(valid_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, test_loader, valid_loader, scaler


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, pre_len, hidden_size, n_layers, dropout=0.05):
        super(LSTM, self).__init__()
        self.pre_len = pre_len
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.hidden = nn.Linear(input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, n_layers, bias=True,
                            batch_first=True)  # output (batch_size, obs_len, hidden_size)
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        batch_size, obs_len, features_size = x.shape  # (batch_size, obs_len, features_size)
        xconcat = self.hidden(x)  # (batch_size, obs_len, hidden_size)
        H = torch.zeros(batch_size, obs_len - 1, self.hidden_size).to(device)  # (batch_size, obs_len-1, hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(
            device)  # (num_layers, batch_size, hidden_size)
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)  # (batch_size, 1, hidden_size)
            out, (ht, ct) = self.lstm(xt, (ht, ct))  # ht size (num_layers, batch_size, hidden_size)
            htt = ht[-1, :, :]  # (batch_size, hidden_size)
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)  # (batch_size, obs_len-1, hidden_size)
        x = self.linear(H)
        return x[:, -self.pre_len:, :]


def train(model, args, scaler, device):
    start_time = time.time()  # 计算起始时间
    model = model
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = args.epochs
    model.train()  # 训练模式
    results_loss = []
    for i in tqdm(range(epochs)):
        losss = []
        for seq, labels in train_loader:
            optimizer.zero_grad()

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)

            single_loss.backward()

            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())
        tqdm.write(f"\t Epoch {i + 1} / {epochs}, Loss: {sum(losss) / len(losss)}")
        results_loss.append(sum(losss) / len(losss))

        torch.save(model.state_dict(), './model/save_model.pth')
        time.sleep(0.1)

    # valid_loss = valid(model, args, scaler, valid_loader)
    # 尚未引入学习率计划后期补上
    # 保存模型

    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")
    plot_loss_data(results_loss)


def valid(model, args, scaler, valid_loader):
    lstm_model = model
    # 加载模型进行预测
    lstm_model.load_state_dict(torch.load('./model/save_model.pth'))
    lstm_model.eval()  # 评估模式
    losss = []

    for seq, labels in valid_loader:
        pred = lstm_model(seq)
        mae = calculate_mae(pred.detach().numpy().cpu(), np.array(labels.detach().cpu()))  # MAE误差计算绝对值(预测值  - 真实值)
        losss.append(mae)

    print("验证集误差MAE:", losss)
    return sum(losss) / len(losss)


def test(model, args, test_loader, scaler):
    # 加载模型进行预测
    losss = []
    model = model
    model.load_state_dict(torch.load('./model/save_model.pth'))
    model.eval()  # 评估模式
    results = []
    labels = []
    for seq, label in test_loader:
        pred = model(seq)
        mae = calculate_mae(pred.detach().cpu().numpy(),
                            np.array(label.detach().cpu()))  # MAE误差计算绝对值(预测值  - 真实值)
        losss.append(mae)
        pred = pred[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])

    print("测试集误差MAE:", losss)
    # # 绘制历史数据
    # plt.plot(labels, label='TrueValue')
    #
    # # 绘制预测数据
    # # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    # plt.plot(results, label='Prediction')
    #
    # # 添加标题和图例
    # plt.title("test state")
    # plt.legend()
    # plt.show()


# 检验模型拟合情况
def inspect_model_fit(model, args, train_loader, scaler):
    model = model
    model.load_state_dict(torch.load('./model/save_model.pth'))
    model.eval()  # 评估模式
    results = []
    labels = []

    for seq, label in train_loader:
        pred = model(seq)[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])

    # # 绘制历史数据
    # plt.plot(labels, label='History')
    #
    # # 绘制预测数据
    # # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    # plt.plot(results, label='Prediction')
    #
    # # 添加标题和图例
    # plt.title("inspect model fit state")
    # plt.legend()
    # plt.show()


def predict(model, args, device, scaler):
    # 预测未知数据的功能
    df = pd.read_csv(args.data_path)
    df = df.iloc[:, 1:][-args.window_size:].values  # 转换为nadarry
    pre_data = scaler.transform(df)
    tensor_pred = torch.FloatTensor(pre_data).to(device)
    tensor_pred = tensor_pred.unsqueeze(0)  # 单次预测 , 滚动预测功能暂未开发后期补上
    model = model
    model.load_state_dict(torch.load('./model/save_model.pth'))
    model.eval()  # 评估模式

    pred = model(tensor_pred)[0]

    pred = scaler.inverse_transform(pred.detach().cpu().numpy())

    # 假设 df 和 pred 是你的历史和预测数据

    # 计算历史数据的长度
    history_length = len(df[:, -1])

    # 为历史数据生成x轴坐标
    history_x = range(history_length)

    # 为预测数据生成x轴坐标
    # 开始于历史数据的最后一个点的x坐标
    prediction_x = range(history_length - 1, history_length + len(pred[:, -1]) - 1)

    # 绘制历史数据
    plt.plot(history_x, df[:, -1], label='History')

    # 绘制预测数据
    # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    plt.plot(prediction_x, pred[:, -1], label='Prediction')
    # plt.axvline(history_length - 1, color='red')  # 在图像的x位置处画一条红色竖线
    # 添加标题和图例
    plt.title("History and Prediction")
    plt.legend()
    plt.savefig('forecast.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='LSTM', help="模型持续更新")
    parser.add_argument('-window_size', type=int, default=144, help="时间窗口大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=12, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
    parser.add_argument('-data_path', type=str, default='./data/processed_ali20_c.csv', help="你的数据数据地址")
    parser.add_argument('-target', type=str, default='avgmem', help='你需要预测的特征列，这个值会最后保存在csv文件里')
    parser.add_argument('-input_size', type=int, default=2, help='你的特征个数不算时间那一列')
    parser.add_argument('-feature', type=str, default='M', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')

    # learning
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.05, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=4, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=16, help="批次大小")
    parser.add_argument('-save_path', type=str, default='models')

    # model
    parser.add_argument('-hidden_size', type=int, default=64, help="隐藏层单元数")
    parser.add_argument('-kernel_sizes', type=int, default=3)
    parser.add_argument('-laryer_num', type=int, default=2)
    # device
    parser.add_argument('-use_gpu', type=bool, default=False)
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

    # option
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)
    parser.add_argument('-lr-scheduler', type=bool, default=True)
    args = parser.parse_args()

    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    print("使用设备:", device)
    train_loader, test_loader, valid_loader, scaler = create_dataloader(args, device)

    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size

    # 实例化模型
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = LSTM(args.input_size, args.output_size, args.pre_len, args.hidden_size, args.laryer_num,
                     args.drop_out).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    # 训练模型
    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, scaler, device)
    if args.test:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        test(model, args, test_loader, scaler)
    if args.inspect_fit:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始检验{args.model}模型拟合情况<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        inspect_model_fit(model, args, train_loader, scaler)
    if args.predict:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        predict(model, args, device, scaler)