import math
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
import h5py
import os
import random
import scipy.io as scio
from Model_define_pytorch import AutoEncoder

#gpu_list = '0'
#os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
SEED = 42
seed_everything(SEED)
mean = 0.50
std = 0.01

import scipy.io as scio
data_load_address = 'train'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
x_train = mat['H_train']  # shape=8000*126*128*2

x_train = np.transpose(x_train.astype('float32'),[0,3,1,2])
print(np.shape(x_train))  # 8000 * 2 *126 *128

mat = scio.loadmat(data_load_address+'/Htest.mat')
x_test = mat['H_test']  # shape=2000*126*128*2

x_test = np.transpose(x_test.astype('float32'),[0,3,1,2])
print(np.shape(x_test))
def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def NMSE_cuda(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x), -1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x), -1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real ** 2 + x_imag ** 2, axis=1)
    mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
    nmse = mse / power
    return nmse
 
class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction
 
    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse


def Score(NMSE):
    score = 1 - NMSE
    return score


    
class DatasetFolder(Dataset):
    def __init__(self, matData, phase='val'):
        self.matdata = matData
        self.phase = phase
        self.data_shape = matData[0].shape
 
    def __getitem__(self, index):
        y = self.matdata[index]
        return y
 
    def __len__(self):
        return self.matdata.shape[0]

class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]
batch_size = 32
epochs = 1000
learning_rate = 2e-3
name = "baseline_no_cluster"
feedback_bits = 512
best_loss =100
model = AutoEncoder(feedback_bits,efn_name='tf_efficientnet_b0_ns')
criterion = NMSELoss(reduction='mean')  # nn.MSELoss()
criterion_test = NMSELoss(reduction='sum')

train_dataset = DatasetFolder(x_train,phase = 'train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=0)
test_dataset = DatasetFolder(x_test,phase = 'val')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,num_workers=0)
for d in train_loader:
    print(d.shape)
    break
model = model.cuda()
import time 
mse_fn = nn.MSELoss()
learning_rate = 3e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.05,betas=(0.9,0.95))
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,T_max=50,T_warmup=3 ,eta_min=5e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5,  min_lr=5e-9, verbose=False)
for epoch in range(epochs):
    print('========================lr:%.4e' % optimizer.param_groups[0]['lr'])
    # 训练
    model.train()
    #if epoch % 100 == 0 and epoch > 0:
        #optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    running_loss = 0.0
    start_time = time.time()
    for i, input in enumerate(train_loader):
        input = input.cuda()
        output= model(input)
 
        loss = mse_fn(output, input)
        loss2 = criterion(output,input)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.8f}\t  Loss2: {loss2:.8f}'.format(
                epoch, i, len(train_loader), loss=loss.item(),loss2=loss2.item()))
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    #r.log_ml(epoch=epoch, loss=epoch_loss)
    # 验证
    print('time for this epch:',time.time()-start_time)
    model.eval()
    total_loss = 0
    mse_loss = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # convert numpy to Tensor
            input = input.cuda()
            output = model(input)
            total_loss += criterion_test(output, input).item()
            mse_loss += mse_fn(output, input).item()
        average_loss = total_loss / len(test_dataset)
        average_loss_mse = mse_loss
        print('NMSE %.4f' % average_loss,'  MSE %.10f' % average_loss_mse)
        if average_loss < best_loss:
            # model save
            # save encoder
            modelSave1 = './models/encoder.pth.tar'
            torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
            # save decoder
            modelSave2 ='./models/decoder.pth.tar'
            torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            print('Model saved!')
            best_loss = average_loss
    scheduler.step(average_loss)