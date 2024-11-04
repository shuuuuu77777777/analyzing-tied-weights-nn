import numpy as np
import torch
import torchvision
import torch.nn as nn
import math
import time
import logging
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
import os
import sys
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset
from torch import optim
from tqdm import tqdm
from pathlib import Path

from data_loading import data_loading
from data_loading import data_loading_test
from my_loss import mse_loss,PSNR


from PnPnet import PnP_Net
from make_kernel import make_gaussblurk
from make_kernel import make_squarek
from tie_weight import tie_process

from denoiser import denoise_grad_o_for_2d



# データの前処理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# トイデータのパラメータ
num_samples = 100000  # サンプル数


# 2つの特徴量を持つデータの生成
x = torch.rand(num_samples, 1)  # 1要素目を持つランダムなサンプル
x = torch.cat((x, x), dim=1)  # 1要素目を2要素目にコピー

y = x.clone()  # yはクリーンなサンプル



class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# データローダの準備
dataset = SimpleDataset(x, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
 

# # モデルの定義
# class denoise_grad_o_for_2d(nn.Module):
#     def __init__(self, n_in=2, n_out=2):
#         super().__init__()
#         self.encoder = nn.Linear(n_in, 50)
#         self.act = nn.ReLU()
#         self.decoder = nn.Linear(50, n_out)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.act(x)
#         with torch.no_grad():
#             self.decoder.weight.copy_(self.encoder.weight.T)
#         x = self.decoder(x)
#         return x
    


# モデル、損失関数、オプティマイザの初期化
model = denoise_grad_o_for_2d()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
noise_level = 0.05

# ノイズを追加する関数
def add_noise(img):
    noise = torch.randn(img.size()) * noise_level  # img と同じ形状のノイズを生成
    noisy_img = img + noise
    return noisy_img



epoch_num = []
loss_ = []



# モデルの学習
num_epochs = 100
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data  # img の形状は (バッチサイズ, 2) になるはず
        
        noisy_img = add_noise(img)  # ノイズを追加
        output = model.forward(noisy_img)  # モデルに渡す


        # output = model(img)  # モデルに渡す
        
        loss = criterion(output, img)  # 損失を計算
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epoch_num.append(epoch)
    loss_.append(loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



# # PSNRをプロット
# plt.figure(figsize=(10, 5))
# plt.plot(epoch_num, loss_, marker='o')
# plt.title('epoch/loss')
# plt.xlabel('epoch')
# plt.ylabel('MSEloss')
# plt.grid(True)
# plt.show()



# 学習後にエンコーダーとデコーダーの重みを保存
torch.save(model.encoder.weight.data, 'encodernoiz0_weights.pth')
torch.save(model.decoder.weight.data, 'decodernoiz0_weights.pth')





