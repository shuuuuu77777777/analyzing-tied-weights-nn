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
import torchvision.transforms as transforms
from PIL import Image

from scipy.linalg import pinv
import cvxpy as cp


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


#条件
net_name = "gradNet_onelayer"
kernel = "noise_only"
weight_tie = True
noise_lev = 0.1
gray = True
test_path = "/home/daito/analyzing-tied-weights-nn/BSDS300/images/test"


crop_size = 3


kernel_size = 3
kernel = torch.tensor([])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの初期化
model = PnP_Net(net_name, kernel, crop_size, device)
picture_num = 0
print("いい感じ")

#cropsize
n = crop_size


#checkpoint_path ="C:\\Users\\toxic\\Desktop\\WGW\\aa\\para_debug\\train_denoiser3\\gradNet_onelayer\\11_gradNet_onelayer_noise=0.1_tie_epoch_360.pth" #BSDS300 hardtanh
# checkpoint_path ="C:\\Users\\toxic\\Desktop\\WGW\\aa\\para_debug\\train_denoiser3\\gradNet_onelayer\\36_gradNet_onelayer_noise=0.1_tie_epoch_40.pth" #MNIST hardtanh

#checkpoint_path = "~/analyzing-tied-weights-nn/para_debug/train_denoiser3/gradNet_onelayer/16_gradNet_onelayer_noise=0.1_tie_epoch_360.pth" #relu
checkpoint_path = "/home/daito/analyzing-tied-weights-nn/para_debug/train_denoiser3/gradNet_onelayer/38_gradNet_onelayer_noise=0.1_tie_epoch_160.pth"

# checkpoint_path = "C:\\Users\\toxic\\Desktop\\WGW\\aa\\encoder4_weights.pth"



# チェックポイントのロード
checkpoint = torch.load(checkpoint_path)

# print(checkpoint)





# モデルの重み全体を表示
for key, value in checkpoint.items():
    print(f"Layer: {key}")
    print(value)  # 各層の重みを表示

kernel_vector = value.cpu().numpy()
kernel_vector = kernel_vector[0, 0]
print(kernel_vector)

# kernel_vector = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(kernel_vector)







# 例として5×5行列
W_conv = np.zeros((crop_size*crop_size, crop_size*crop_size))



for i in range(crop_size):
    for j in range(crop_size):
        if 0<i<n-1 and 0<j<n-1:  ##海なし
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1-1] = kernel_vector[0,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1] = kernel_vector[0,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1] = kernel_vector[0,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1-1] = kernel_vector[1,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1] = kernel_vector[1,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1] = kernel_vector[1,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1-1] = kernel_vector[2,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1] = kernel_vector[2,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1] = kernel_vector[2,2]

        if 0<i<n-1  and j == 0:  ##左
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1-1] = kernel_vector[0,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1] = kernel_vector[0,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1] = kernel_vector[0,2]
        
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1-1] = kernel_vector[1,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1] = kernel_vector[1,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1] = kernel_vector[1,2]
        
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1-1] = kernel_vector[2,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1] = kernel_vector[2,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1] = kernel_vector[2,2]

        if 0<i<n-1 and j ==n-1 :  ##右
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1-1] = kernel_vector[0,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1] = kernel_vector[0,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1] = kernel_vector[0,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1-1] = kernel_vector[1,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1] = kernel_vector[1,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1] = kernel_vector[1,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1-1] = kernel_vector[2,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1] = kernel_vector[2,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1] = kernel_vector[2,2]

        if i == 0 and 0<j<n-1:  ##上
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1-1] = kernel_vector[0,0]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1] = kernel_vector[0,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1] = kernel_vector[0,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1-1] = kernel_vector[1,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1] = kernel_vector[1,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1] = kernel_vector[1,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1-1] = kernel_vector[2,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1] = kernel_vector[2,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1] = kernel_vector[2,2]

        if i == n-1 and 0<j<n-1: ##下
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1-1] = kernel_vector[0,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1] = kernel_vector[0,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1] = kernel_vector[0,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1-1] = kernel_vector[1,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1] = kernel_vector[1,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1] = kernel_vector[1,2]
        
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1-1] = kernel_vector[2,0]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1] = kernel_vector[2,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1] = kernel_vector[2,2]


        if i == 0 and j == 0:  ##左上
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1-1] = kernel_vector[0,0]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1] = kernel_vector[0,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1] = kernel_vector[0,2]
        
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1-1] = kernel_vector[1,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1] = kernel_vector[1,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1] = kernel_vector[1,2]
        
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1-1] = kernel_vector[2,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1] = kernel_vector[2,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1] = kernel_vector[2,2]

        if i == 0 and j == n-1:  ##右上
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1-1] = kernel_vector[0,0]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1] = kernel_vector[0,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1] = kernel_vector[0,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1-1] = kernel_vector[1,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1] = kernel_vector[1,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1] = kernel_vector[1,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1-1] = kernel_vector[2,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1] = kernel_vector[2,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1] = kernel_vector[2,2]

        if i == n-1 and j == 0:  ##左下
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1-1] = kernel_vector[0,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1] = kernel_vector[0,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1] = kernel_vector[0,2]
        
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1-1] = kernel_vector[1,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1] = kernel_vector[1,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1] = kernel_vector[1,2]
        
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1-1] = kernel_vector[2,0]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1] = kernel_vector[2,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1] = kernel_vector[2,2]

        if i == n-1 and j == n-1:  #右下
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1-1] = kernel_vector[0,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1] = kernel_vector[0,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1] = kernel_vector[0,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1-1] = kernel_vector[1,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1] = kernel_vector[1,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1] = kernel_vector[1,2]
        
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1-1] = kernel_vector[2,0]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1] = kernel_vector[2,1]
            # W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1] = kernel_vector[2,2]


        
        


        




################# load image data and apply

def prepare_data(device, test_path, noise_lev, gray, crop_size):
    test_dataset = data_loading_test(device, model.grad.H,test_path, noise_lev, gray, crop_size)
    test_dataloader = DataLoader(test_dataset, batch_size=10)  # 単一サンプルで評価
    return test_dataloader

# データの準備
test_dataloader = prepare_data(device, test_path, noise_lev, gray, crop_size)


# テストデータの中から、最初のバッチを取得
for batch in test_dataloader:
    noisy_image = batch["noise"][picture_num,:,:,:]  # リストから最初の要素を取得
    original_image = batch["true"][picture_num,:,:,:]  # リストから最初の要素を取得
    break  # 1つ目のバッチだけを使用する場合


print("いい感じ")


denoised_image = noisy_image.clone()



#####################################



# 画像を行ごとに縦ベクトルに変換する関数
def image_to_column_vector(image_tensor):
    # 画像テンソルの次元を (H, W) に変更
    image_tensor = image_tensor.squeeze(0)  # (1, H, W) -> (H, W)
    
    # 各行を縦ベクトルに並べる
    column_vector = image_tensor.view(-1, 1)  # (H * W, 1)
    
    return column_vector

# 画像テンソルから縦ベクトルに変換
column_vector = image_to_column_vector(noisy_image)
#column_vector = image_to_column_vector(original_image)




# Convert column_vector to a numpy array
column_vector_np = column_vector.cpu().numpy()  # Ensure it's on CPU for numpy


# column_vector_np が想定の形状でない場合 reshape
column_vector_np = column_vector_np.reshape(-1, 1)
#print("column_vector_np shape (after reshape):", column_vector_np.shape)




##計算]
print(W_conv)
m = crop_size*crop_size
I_m = np.eye(m)

W_conv_T_t= pinv(W_conv.T)
print('いい感じ')

# 最適化変数
k = cp.Variable(9)
print(k)

A = I_m - W_conv_T_t@W_conv.T


print('A',A.shape)
b = -W_conv_T_t@column_vector_np
print('b',b.shape)


# 目的関数 (最小二乗法)
objective = cp.Minimize(0.5 * cp.sum_squares(A @ k - b))
print('いい感じ1')
# 制約条件
constraints = [A @ k >= b]
print('いい感じ2')
# 問題の定義
problem = cp.Problem(objective, constraints)
print('いい感じ3')


# 解を求める
problem.solve()
print('いい感じ4')

if k.value is not None:
    output = A @ k.value - b
else:
    print("k.value is None. Optimization failed.")

output = A@k.value - b


####################################################
# ##denoiser forwardの再現
# output =W_conv.T@ column_vector_np 
# for i in range(crop_size):
#     if output[i] < 0:
#         output[i] = 0
# output = W_conv@output
###################################################



# ベクトルを画像行列に変換
matrix = output.reshape(crop_size, crop_size)
#print(matrix)


print("いい感じ")






##画像の表示


base_dir = os.path.expanduser("~/analyzing-tied-weights-nn")
save_dir = os.path.join(base_dir, 'img')
os.makedirs(save_dir, exist_ok=True)

import matplotlib.colors as mcolors


#red&gray
# データの最小値と最大値を取得
vmin, vmax = np.min(matrix), np.max(matrix)

# 負の値を赤色、非負の値をグレースケールで表示するためにカスタムカラーマップを定義
cmap = mcolors.ListedColormap(['red', 'gray'])

# カラーマップの境界をデータ範囲に基づいて設定
bounds = [vmin, 0, vmax]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# # 画像を生成し、ファイルに保存
plt.imshow(matrix, cmap=cmap, norm=norm)
# plt.colorbar()  # カラーバーを追加
plt.savefig(os.path.join(save_dir, 'WTty_NN_with_noise.png'))
plt.close()


# #グレースケール
# # # 画像を生成し、ファイルに保存
# plt.imshow(matrix, cmap='gray')
# plt.colorbar()  # カラーバーを追加
# plt.savefig(os.path.join(save_dir, 'WTty_NN_with_noise.png'))
# plt.close()