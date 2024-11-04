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
from denoiser import denoise_grad_o_for_2d

from PnPnet import PnP_Net
from make_kernel import make_gaussblurk
from make_kernel import make_squarek
#from tie_weight import tie_process



# モデルの初期化
model = denoise_grad_o_for_2d()

# 保存された重みを読み込む
encoder_weights = torch.load('encodernoiz0_weights.pth')
decoder_weights = torch.load('decodernoiz0_weights.pth')



# # 重みが正しく読み込まれているか確認
print(torch.equal(model.encoder.weight))
# print(torch.equal(model.decoder.weight, torch.load('decodernoiz0_weights.pth')['weight']))





# モデルに重みを設定
model.encoder.weight.data.copy_(encoder_weights)
model.decoder.weight.data.copy_(decoder_weights)



x = torch.rand(1, 1)  # 1要素目を持つランダムなサンプル
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
test_loader = DataLoader(dataset, batch_size=32, shuffle=True)



noise_level =0.01

# ノイズを追加する関数
def add_noise(img):
    noise = torch.randn(img.size()) * noise_level  # img と同じ形状のノイズを生成
    noisy_img = img + noise
    return noisy_img

# 出力画像の生成 (バッチ処理に変更)
test_data_iter = iter(test_loader)
test_images, _ = next(test_data_iter)  # テストデータの取得
test_images = test_images.view(test_images.size(0), -1)  # フラット化
noisy_test_images = add_noise(test_images)  # ノイズを追加

# バッチ全体に対して推論を行う
output_images = model(noisy_test_images)

# 複数の画像を扱う場合はループで処理
for index in range(test_images.size(0)):
    original_img = test_images[index].detach()
    noisy_img = noisy_test_images[index].clone().detach()
    output_img = output_images[index].detach()
    
    # ここでPSNRや他の評価を実施

######
# # 出力画像の生成
# test_data_iter = iter(test_loader)
# test_images, _ = next(test_data_iter)
# test_images = test_images.view(test_images.size(0), -1)
# noisy_test_images = add_noise(test_images)
# output_images = model(noisy_test_images)

# # 初期入力としてノイズ付きテスト画像を使用
# index = 0  # 表示する画像のインデックス
# original_img = test_images[index].detach()
# noisy_img = noisy_test_images[index].clone().detach()

# # イテレーションプロセス
# input_img = noisy_img.clone().detach()
# output_img = noisy_img.clone().detach()

# print(output_img)

########



# #output_img = add_noise(x)
# output_img = add_noise(x)

# print(output_img.unsqueeze(0))
# input_img = output_img.clone()


# パラメータ設定
lambda_ = 0.8 # λの値を設定（必要に応じて調整）

# イテレーションプロセス
num_iterations = 1



for i in range(num_iterations):    
    # x_k - λ * ∇f(x_k) を計算
    #output_img = output_img - lambda_ * (output_img -input_img.detach())
    
    # モデルで更新
    output_img = model.forward(output_img.unsqueeze(0)).squeeze().detach()
    
    # PSNRを計算
    # psnr = calculate_psnr(original_img, output_img)
    # psnr_values.append(psnr)
    #print(f"Iteration {i+1}/{num_iterations}, PSNR: {psnr:.4f} dB")



# # プロット
# plt.figure(figsize=(8, 6))
# plt.scatter(x.numpy(), y.numpy(), label='データ点', color='blue')
# plt.plot(x.numpy(), x.numpy(), label='y = x', color='red', linestyle='--')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('y = x を平均値としたトイデータ')
# plt.legend()
# plt.grid()
# plt.show()




print(output_img)