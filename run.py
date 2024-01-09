import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from networks import Q_net, DeepSurv, D_net_gauss, NegativeLogLikelihood
from utils import c_index
from utils import read_config
from torch.utils.data import TensorDataset, DataLoader, Dataset

import argparse

parser = argparse.ArgumentParser(description='train AAE model')
parser.add_argument('--z_dim', type=int, default=50, help='size of the latent z vector')
parser.add_argument('--batch_size', type=int, default=32, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--data_size', type=int, default=11376, help='number of data_size')
parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--select_model", type=int, default=0, help="a type of models in five DeepSurv models")
opt = parser.parse_args([])
print(opt)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载数据集
ss = MinMaxScaler()
time_ss = MinMaxScaler()
# data = pd.read_csv('gene_data/blca.csv', index_col='DATA')
# data = pd.read_csv('./gene_data/gse13507_co3_output.csv', index_col='DATA')
# data = pd.read_csv('./gene_data/gse31684_co3_output.csv', index_col='DATA')
data = pd.read_csv('./gene_data/gse32894_co3_output.csv', index_col='DATA')

print(data)
data_feature = data.iloc[:, :-2]
data_label = data.iloc[:, -1]
data_death = data.iloc[:, -1]
data_time = data.iloc[:, -2]
# 归一化输入
std_data_feature = ss.fit_transform(data_feature.values)
data_time = time_ss.fit_transform(np.array(data_time).reshape(-1, 1))
data_time = np.array(data_time).reshape(-1, 1)
data_death = np.array(data_death).reshape(-1, 1)

tensor_data = torch.Tensor(np.concatenate([std_data_feature, data_time, data_death], axis=1))
print(tensor_data.shape)

params = [
    ('Simulated Linear', 'linear.ini'),
    ('Simulated Nonlinear', 'gaussian.ini'),
    ('WHAS', 'whas.ini'),
    ('SUPPORT', 'support.ini'),
    ('METABRIC', 'metabric.ini'),
    ('Simulated Treatment', 'treatment.ini'),
    ('Rotterdam & GBSG', 'gbsg.ini')]

name = params[opt.select_model][0]
ini_file = params[opt.select_model][1]

config = read_config('configs/' + ini_file)

config['train'].pop('h5_file')

config['network']['dims'] = [opt.z_dim, 1000, 1]

EPS = 1e-15
# Set learning rates
gen_lr = 0.0001
reg_lr = 0.00005
deep_Surv_lr = 0.0001

# Set models and optimizers
DeepSurv_net = DeepSurv(config['network']).to(DEVICE)
Q = Q_net(opt.data_size, 1000, opt.z_dim).to(DEVICE)

# import model
#checkpoint = torch.load("models/model.pth")
# DeepSurv_net.load_state_dict(checkpoint['DeepSurv_net'])
# Q.load_state_dict(checkpoint['Q'])


def valid_fuc(dataloader, deepSurv_model, encoder_model):
    deepSurv_model.eval()
    encoder_model.eval()
    # criterion_risk = NegativeLogLikelihood(0).to(DEVICE)
    for data_batch in dataloader:
        X = data_batch[:, :-2].to(DEVICE)
        y = data_batch[:, -2].to(DEVICE)
        e = data_batch[:, -1].to(DEVICE)
        enc_X = encoder_model(X).to(DEVICE)
        risk_pred = deepSurv_model(enc_X)
        # risk_loss = criterion_risk(risk_pred, y, e, deepSurv_model)
        valid_c = c_index(-risk_pred, y, e)
        return valid_c, risk_pred


# c_index_list=[]
# for data_batch in tensor_data:
risk_list = []
tensor_data_loader = DataLoader(tensor_data, batch_size=len(tensor_data), shuffle=True)
current_c_index, current_risk_pred = valid_fuc(tensor_data_loader, DeepSurv_net, Q)
print(current_c_index)
# print(current_risk_pred)
current_risk_pred = current_risk_pred.to('cpu')
risk_pred_array = current_risk_pred.detach().numpy()
df = pd.DataFrame(risk_pred_array)
# df.to_csv('risk_gse13507.csv', index=False)
# df.to_csv('risk_gse31684.csv', index=False)
df.to_csv('risk_gse32894.csv', index=False)

# a = pd.DataFrame(c_index_list)
# a.to_csv('c_index_final_blca1.csv')
# a.to_csv('c_index_final01.csv')
# a.to_csv('c_index_final02.csv')
# a.to_csv('c_index_final03.csv')
# a.to_csv('c_index_final04.csv')
# a.to_csv('c_index_final05.csv')
