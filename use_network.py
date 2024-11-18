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

from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from pathlib import Path

from data_loading import data_loading
from data_loading import data_loading_test
from torch.utils.data import Dataset, DataLoader
from my_loss import mse_loss,PSNR

from PnPnet import PnP_Net
from make_kernel import make_gaussblurk
from make_kernel import make_squarek
from tie_weight import tie_process







def load_model_weights(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)




def prepare_data(device, test_path, noise_lev, gray, crop_size):
    test_dataset = data_loading_test(device, model.grad.H,test_path, noise_lev, gray, crop_size)
    test_dataloader = DataLoader(test_dataset, batch_size=10)  # 単一サンプルで評価
    return test_dataloader





#設定
device = torch.device("cuda")

net_name = "gradNet_onelayer"
kernel = "noise_only"
weight_tie = True

#########BSDS
crop_size = 64
test_path = "/home/daito/analyzing-tied-weights-nn/BSDS300/images/test"


# ######MNIST
# crop_size = 28
# test_path = "/home/daito/analyzing-tied-weights-nn/MNIST Dataset JPG format/MNIST - JPG - testing/0"



checkpoint_path ="/home/daito/analyzing-tied-weights-nn/para_debug/train_denoiser3/gradNet_onelayer/38_gradNet_onelayer_noise=0.1_tie_epoch_160.pth"



noise_lev = 0.1
gray = True





# モデルの初期化

kernel_size = 1
kernel = torch.tensor([])

#model = get_model(net_name, kernel, gray, weight_tie, crop_size, device)
model = PnP_Net(net_name, kernel, crop_size, device, 1, weight_tie)


# 保存した重みのパス（例: checkpoint_path_debug/ラン名.pth）
model_path = checkpoint_path


# 重みをロードしてモデルに適用
state_dict = torch.load(model_path)
model.denoiser.load_state_dict(state_dict)





# # 重みをロード
# state_dict = torch.load(checkpoint_path)

# # キー名を修正
# from collections import OrderedDict
# new_state_dict = OrderedDict()

# for key, value in state_dict.items():
#     new_key = key if key.startswith("denoiser.") else "denoiser." + key
#     new_state_dict[new_key] = value

# # 修正したキーで重みを適用
# model.load_state_dict(new_state_dict)





# モデルを評価モードに設定（学習時のdropoutやbatchnormなどを無効にするため）
model.eval()




# データの準備
test_dataloader = prepare_data(device, test_path, noise_lev, gray, crop_size)




# テストデータの中から、最初のバッチを取得
for batch in test_dataloader:
    noisy_image = batch["noise"][0,:,:,:]  # リストから最初の要素を取得
    original_image = batch["true"][0,:,:,:]  # リストから最初の要素を取得
    break  # 1つ目のバッチだけを使用する場合




#iteration 
iteration_num = 1

psnr_array = []

denoised_image = noisy_image.clone()  # ノイズのある画像を初期値にする
#denoised_image = original_image.clone() #

for i in range(iteration_num):
    # PSNRを計算
    psnr_value = PSNR(denoised_image, original_image)
    print(psnr_value)
    # PSNRの値を保存
    psnr_array.append(psnr_value)

    # モデルでデノイジング
    #denoised_image = model.forward(denoised_image, noisy_image,0.7)
    denoised_image = model.forward(denoised_image, noisy_image)
    print(PSNR(denoised_image, original_image))
    
# print(psnr_array)

denoised_image = torch.clamp(denoised_image,0,1)
    


# 画像の表示
noisy_image = noisy_image.squeeze().cpu().numpy()
original_image = original_image.squeeze().cpu().numpy()
denoised_image = denoised_image.squeeze().cpu().detach().numpy()




# print(psnr_array)


# plt.figure(figsize=(12, 3))

# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(original_image, vmin=0, vmax=1,cmap='gray' if gray else None)

# plt.subplot(1, 3, 2)
# plt.title('Noisy Image')
# plt.imshow(noisy_image, vmin=0, vmax=1,cmap='gray' if gray else None)

# plt.subplot(1, 3, 3)
# plt.title('Denoised Image')
# plt.imshow(denoised_image,vmin=0, vmax=1, cmap='gray' if gray else None)
# plt.show()

# # PSNRのプロットは別の図で作成
# plt.figure()
# plt.plot(range(iteration_num), psnr_array)
# plt.xlabel('Iteration')
# plt.ylabel('PSNR')
# plt.title('PSNR over Iterations')
# plt.show()

# # PSNR差分を計算
# psnr_diff = [abs(psnr_array[i+1] - psnr_array[i]) for i in range(len(psnr_array)-1)]

# # PSNRの差分をプロット
# plt.figure()
# plt.semilogy(range(0, iteration_num-1), psnr_diff)  # 差分はiteration 1から開始
# plt.xlabel('Iteration')
# plt.ylabel('PSNR Difference')
# plt.title('Difference in PSNR between Consecutive Iterations')
# plt.show()




import os


# 保存先ディレクトリを設定
base_dir = os.path.expanduser("~/analyzing-tied-weights-nn")
save_dir = os.path.join(base_dir, 'img')
os.makedirs(save_dir, exist_ok=True)


plt.figure(figsize=(12, 3))

# Original Image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(original_image, vmin=0, vmax=1, cmap='gray' if gray else None)

# Noisy Image
plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisy_image, vmin=0, vmax=1, cmap='gray' if gray else None)

# Denoised Image
plt.subplot(1, 3, 3)
plt.title('Denoised Image')
plt.imshow(denoised_image, vmin=0, vmax=1, cmap='gray' if gray else None)

# 画像をファイルとして保存
plt.savefig(os.path.join(save_dir, 'images_comparison.png'))
plt.close()  # メモリ節約のためにプロットを閉じる

# # PSNRのプロットは別の図で作成
# plt.figure()
# plt.plot(range(iteration_num), psnr_array)
# plt.xlabel('Iteration')
# plt.ylabel('PSNR')
# plt.title('PSNR over Iterations')

# # PSNRグラフをファイルとして保存
# plt.savefig(os.path.join(save_dir, 'psnr_over_iterations.png'))
# plt.close()

# # PSNR差分を計算
# psnr_diff = [abs(psnr_array[i+1] - psnr_array[i]) for i in range(len(psnr_array)-1)]

# # PSNRの差分をプロット
# plt.figure()
# plt.semilogy(range(0, iteration_num-1), psnr_diff)  # 差分はiteration 1から開始
# plt.xlabel('Iteration')
# plt.ylabel('PSNR Difference')
# plt.title('Difference in PSNR between Consecutive Iterations')

# # PSNR差分グラフをファイルとして保存
# plt.savefig(os.path.join(save_dir, 'psnr_difference.png'))
# plt.close()


