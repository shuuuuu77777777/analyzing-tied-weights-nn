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

#cropsize
n = 28



#checkpoint_path ="C:\\Users\\toxic\\Desktop\\WGW\\aa\\para_debug\\train_denoiser3\\gradNet_onelayer\\11_gradNet_onelayer_noise=0.1_tie_epoch_360.pth" #BSDS300 hardtanh
checkpoint_path ="C:\\Users\\toxic\\Desktop\\WGW\\aa\\para_debug\\train_denoiser3\\gradNet_onelayer\\36_gradNet_onelayer_noise=0.1_tie_epoch_40.pth" #MNIST hardtanh

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
# print(np.shape(kernel_vector))





# 例として5×5行列
W_conv = np.zeros((n*n, n*n))


for i in range(n):
    for j in range(n):
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



        if 0<i<n-1<0 and 0<j<n-1:  ##海なし
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1-1] = kernel_vector[0,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1-1] = kernel_vector[0,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1-1)*n+j+1] = kernel_vector[0,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1-1] = kernel_vector[1,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1-1] = kernel_vector[1,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1-1)*n+j+1] = kernel_vector[1,2]
        
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1-1] = kernel_vector[2,0]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1-1] = kernel_vector[2,1]
            W_conv[((i+1-1)*n+j+1)-1,(i+1)*n+j+1] = kernel_vector[2,2]


print(np.array2string(W_conv, formatter={'float_kind': lambda x: f'{x:0.4f}'}))

# ヒートマップのプロット
plt.imshow(W_conv, cmap='viridis', aspect='auto')
plt.colorbar()  # カラーバーを追加
plt.title("Heatmap of W_conv")
plt.show()


# print(np.linalg.pinv(W_conv.T))



# W_WT = W_conv@(W_conv.T)

# W_WT_1 = np.linalg.inv(W_WT)

# W_giji = W_WT_1@W_conv

# print(W_giji)




# image_temp = np.random.rand(1, 9)
# image_temp = image_temp.T
# rank = np.linalg.matrix_rank(W_conv)
# print(rank)

# print(np.dot(W_giji,image_temp))


# Check if the matrix is invertible by calculating the determinant
# det_W_conv = np.linalg.det(W_conv)

# # Output the determinant and check for invertibility
# print("Determinant of W_conv:", det_W_conv)
# if det_W_conv != 0:
#     print("The inverse exists.")
# else:
#     print("The inverse does not exist.")

            

# kernel =  np.random.randint(0, 10, size=(3, 3))

# eig_row = np.zeros(n**2)

# for i in range(3):
#     eig_row[i*n:i*n+3] = kernel[i,:]

# eig_row_fft = np.fft.fft(eig_row)

# print(np.min(np.abs(eig_row_fft)))


# plt.plot(np.abs(eig_row_fft))
# plt.show()