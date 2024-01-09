#!/usr/bin/env python
# coding: utf-8
import os

# In[1]:


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# In[2]:


import torch
import torch.nn as nn

import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from networks import DeepSurv, Q_net, D_net_gauss
from networks import NegativeLogLikelihood
from tqdm import tqdm_notebook, tqdm

np.set_printoptions(suppress=True)
import random
import warnings

warnings.filterwarnings("ignore")

# In[4]:


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

# In[5]:


ss = MinMaxScaler()
time_ss = MinMaxScaler()
data = pd.read_csv('gene_data/blca.csv', index_col='DATA')
# data2 = pd.read_csv('gene_data/gse13507_co3_output.csv', index_col='DATA')
# data = pd.concat([data1, data2], axis=0)
# data = pd.concat([data1, data2], axis=1)
data_feature = data.iloc[:, :-2]
data_label = data.iloc[:, -1]
data_death = data.iloc[:, -1]
data_time = data.iloc[:, -2]

std_data_feature = ss.fit_transform(data_feature.values)
data_time = time_ss.fit_transform(np.array(data_time).reshape(-1, 1))
data_time = np.array(data_time).reshape(-1, 1)
data_death = np.array(data_death).reshape(-1, 1)

tensor_data = torch.Tensor(np.concatenate([std_data_feature, data_time, data_death], axis=1))


train_tensor_data, test_tensor_data = train_test_split(tensor_data, test_size=0.2, random_state=42)

# In[6]:


# Deep_Surve model
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
MODEL_PATH = "models/model.pth"




def valid_fuc(dataloader, deepSurv_model, encoder_model):
    deepSurv_model.eval()
    encoder_model.eval()
    criterion_risk = NegativeLogLikelihood(config['network']).to(DEVICE)
    for data_batch in dataloader:
        X = data_batch[:, :-2].to(DEVICE)
        y = data_batch[:, -2].to(DEVICE)
        e = data_batch[:, -1].to(DEVICE)
        enc_X = encoder_model(X).to(DEVICE)
        risk_pred = deepSurv_model(enc_X)
        risk_loss = criterion_risk(risk_pred, y, e, deepSurv_model)
        valid_c = c_index(-risk_pred, y, e)
        return valid_c, risk_loss.item()


def train(K_flod, train_dataloader, valid_dataloader):
    # network
    Q = Q_net(opt.data_size, 1000, opt.z_dim).to(DEVICE)
    D_gauss = D_net_gauss(500, opt.z_dim).to(DEVICE)
    DeepSurv_net = DeepSurv(config['network']).to(DEVICE)
    criterion_risk = NegativeLogLikelihood(config['network']).to(DEVICE)

    # optimizers
    optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
    optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
    optim_D = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)
    optim_risk = torch.optim.Adam(DeepSurv_net.parameters(), lr=deep_Surv_lr)

    # start training
    c_index_list = []
    valid_loss_list = []
    list_01 = [0] * 5
    best_c_index = -1
    for epoch in tqdm(range(opt.n_epochs)):
        DeepSurv_net.train()
        Q.train()
        D_gauss.train()

        for data_batch in train_dataloader:
            X = data_batch[:, :-2].to(DEVICE)
            y = data_batch[:, -2].to(DEVICE)
            e = data_batch[:, -1].to(DEVICE)

            Q.zero_grad()
            D_gauss.zero_grad()

            # DeepSurv risk loss
            z_sample = Q(X)  # encode to z
            risk_pred = DeepSurv_net(z_sample)
            risk_loss = criterion_risk(risk_pred, y, e, DeepSurv_net)
            risk_loss.backward()
            optim_risk.step()
            optim_Q_enc.step()

            # Discriminator
            Q.eval()
            z_real_gauss = Variable(torch.randn(X.size()[0], opt.z_dim) * 5.).to(DEVICE)
            D_real_gauss = D_gauss(z_real_gauss)
            z_fake_gauss = Q(X)
            D_fake_gauss = D_gauss(z_fake_gauss)
            D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
            D_loss.backward()
            optim_D.step()

            # Generator
            Q.train()
            z_fake_gauss = Q(X)
            D_fake_gauss = D_gauss(z_fake_gauss)
            G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))
            G_loss.backward()
            optim_Q_gen.step()

        # if epoch == 4998:
        #     print(z_sample[:5])
        valid_c, valid_loss = valid_fuc(valid_dataloader, DeepSurv_net, Q)
        c_index_list.append(valid_c)
        valid_loss_list.append(valid_loss)
        if epoch % 100 == 0:
            print('c_index:{:.3f}, valid_loss:{:.3f}, D_loss{:.3f}, risk_loss{:.3f}'.format(valid_c, valid_loss,
                                                                                            D_loss.item(),
                                                                                            risk_loss.item()))
        if best_c_index < valid_c:
            best_c_index = valid_c


            MODEL_PATH_ls = MODEL_PATH+str(K_flod)
            try:
                os.remove(MODEL_PATH_ls)
                print("MODEL_PATH_ls")
            except:
                pass
            torch.save({
                'DeepSurv_net': DeepSurv_net.state_dict(),
                'Q': Q.state_dict(),
                'best_c_index': best_c_index,
                'epoch': epoch},MODEL_PATH_ls)

    return c_index_list, valid_loss_list


FOLD = 5
kf = KFold(n_splits=FOLD, shuffle=True, random_state=39)
c_index_list = []
valid_loss_list = []


for i, (train_index, test_index) in enumerate(kf.split(np.arange(tensor_data.shape[0]), tensor_data)):
    print(str(i + 1), '-' * 50)
    tra = tensor_data[train_index]
    val = tensor_data[test_index]
    train_loader = DataLoader(tra, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(val, batch_size=len(val), shuffle=False)
    current_c_index, current_valid_loss = train(i, train_loader, test_loader)
    c_index_list.append(current_c_index)
    valid_loss_list.append(current_valid_loss)

a = pd.DataFrame(c_index_list)

a.to_csv('c_index.csv')
