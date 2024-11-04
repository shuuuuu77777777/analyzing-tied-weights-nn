import torch
import torch.nn as nn
import torch.nn.functional as F

from my_layer import sc_sigmoid_layer
from my_layer import skip_act_sig
from my_layer import shifted_softplus
from my_layer import shifted_Relu
from my_layer import soft_shrink
from my_layer import soft_shrink_fix

from my_layer import firm_shrink


from collections import OrderedDict


def requires_grad_False(model):
    if isinstance(model, nn.Conv2d):
        for param in model.parameters():
            param.requires_grad = False




###################################################################################
class denoise_grad_o(nn.Module):
    def __init__(self, n_in=1, n_out=1):
        super().__init__()

        self.inconv = nn.ConvTranspose2d(
            in_channels=n_in, out_channels=1, kernel_size=3, padding=1, bias=False)
        #self.act = nn.ELU()
        #self.act = nn.Hardshrink()#(4)"""
        #self.act = nn.Hardsigmoid()  
        self.act = nn.Hardtanh()
        #self.act = nn.Hardswish()
        #self.act = nn.LeakyReLU()
        #self.act = nn.LogSigmoid()
        ###self.act = nn.MultiheadAttention()
        #self.act = nn.PReLU()
        #self.act = nn.ReLU()
        #self.act = nn.ReLU6()
        #self.act = nn.RReLU()
        #self.act = nn.SELU()
        #self.act = nn.CELU()
        #self.act = nn.GELU()
        #self.act = nn.Sigmoid()
        #self.act = nn.SiLU()
        #self.act = nn.Mish()
        #self.act = nn.Softplus()
        #self.act = nn.Softshrink()
        #self.act = nn.Softsign()
        #self.act = nn.Tanh()
        #self.act = nn.Tanhshrink()
        ########self.act = nn.Threshold()
        #####self.act = nn.GLU()
        #self.act = nn.Softmin()
        #self.act = nn.Softmax()
        #####self.act = nn.Softmax2d()
        #self.act = nn.LogSoftmax()
        #####self.act = nn.AdaptiveLogSoftmaxWithLoss()
        # self.act = 
        # self.act = 
        # self.act = 
        # self.act = 

        self.outconv = nn.Conv2d(
            in_channels=1, out_channels=n_out, kernel_size=3, padding=1, bias=False)
        # if do_fix:
        # self.apply(requires_grad_False)

    def forward(self, x):
        x = self.inconv(x)
        x = self.act(x)
        x = self.outconv(x)
        return x
    
################################################################################

###################################################################################
class denoise_grad_o_for_2d(nn.Module):
    def __init__(self, n_in=1, n_out=1):
        super().__init__()

        self.encoder = nn.Linear(
            2,2)
        #self.act = nn.ELU()
        #self.act = nn.Hardshrink()#(4)"""
        #self.act = nn.Hardsigmoid()  
        #self.act = nn.Hardtanh()
        #self.act = nn.Hardswish()
        #self.act = nn.LeakyReLU()
        #self.act = nn.LogSigmoid()
        ###self.act = nn.MultiheadAttention()
        #self.act = nn.PReLU()
        self.act = nn.ReLU()
        #self.act = nn.ReLU6()
        #self.act = nn.RReLU()
        #self.act = nn.SELU()
        #self.act = nn.CELU()
        #self.act = nn.GELU()
        #self.act = nn.Sigmoid()
        #self.act = nn.SiLU()
        #self.act = nn.Mish()
        #self.act = nn.Softplus()
        #self.act = nn.Softshrink()
        #self.act = nn.Softsign()
        #self.act = nn.Tanh()
        #self.act = nn.Tanhshrink()
        ########self.act = nn.Threshold()
        #####self.act = nn.GLU()
        #self.act = nn.Softmin()
        #self.act = nn.Softmax()
        #####self.act = nn.Softmax2d()
        #self.act = nn.LogSoftmax()
        #####self.act = nn.AdaptiveLogSoftmaxWithLoss()
        # self.act = 
        # self.act = 
        # self.act = 
        # self.act = 


        self.decoder = nn.Linear(2,2)
        


       
        # if do_fix:
        # self.apply(requires_grad_False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.act(x)

# エンコーダの重みの転置をデコーダに設定
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)

        x = self.decoder(x)
        return x
    
############


# import torch.fft


# class FourierTransformLayer(nn.Module):
#     def __init__(self, inverse=False):
#         super(FourierTransformLayer, self).__init__()
#         self.inverse = inverse

#     def forward(self, x):
#         # フーリエ変換を適用
#         if len(x.shape) == 4:  # バッチサイズ、チャネル、高さ、幅
#             # 各チャネルごとにFFTを適用
#             fft = torch.fft.fft2(x)
#             fft = fft.fftshift(fft)  # シフトしてゼロ周波数を中央に
#             magnitude = torch.abs(fft)
#             phase = torch.angle(fft)
#             # 必要に応じて特徴量としてマグニチュードと位相を使用
#             return torch.cat((magnitude, phase), dim=1)
#         else:
#             raise NotImplementedError("Only 4D tensors are supported")

# class SimpleFourierNN(nn.Module):
#     def __init__(self):
#         super(SimpleFourierNN, self).__init__()
#         self.fourier = FourierTransformLayer()
#         self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)  # マグニチュードと位相を入力
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.fc = nn.Linear(32 * 28 * 28, 10)  # 例として10クラス分類

#     def forward(self, x):
#         x = self.fourier(x)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x












class denoise_grad2(nn.Module):
    def __init__(self, n_in=1, n_out=1):
        super().__init__()

        self.alpha = 0  # 活性化関数のシフト
        self.beta = 1  # 活性化関数の係数

        self.inconv = nn.ConvTranspose2d(
            in_channels=n_in, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.act1 = shifted_softplus(alpha=self.alpha, beta=self.beta)
        self.convt1 = nn.ConvTranspose2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False)
        # self.act2 = shifted_softplus(alpha=self.alpha, beta=self.beta)
        self.act2 = sc_sigmoid_layer(beta=self.beta)
        self.conv1 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.act3 = skip_act_sig(alpha=self.alpha, beta=self.beta)
        self.outconv = nn.Conv2d(
            in_channels=16, out_channels=n_out, kernel_size=3, padding=1, bias=False)

        # if do_fix:
        # self.apply(requires_grad_False)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.act1(x1)
        x2 = self.convt1(x2)
        x2 = self.act2(x2)
        x2 = self.conv1(x2)
        x2 = self.act3(x2, x1)
        x2 = x2+x1
        x = self.outconv(x2)
        return x



class denoise_cnn_shifted_relu(nn.Module):
    def __init__(self, n_in=1, n_out=1):
        super().__init__()

        alpha = 0

        self.inconv = nn.ConvTranspose2d(
            in_channels=n_in, out_channels=16, kernel_size=3, padding=1)
        self.act = shifted_Relu(alpha=alpha)
        self.outconv = nn.Conv2d(
            in_channels=16, out_channels=n_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.inconv(x)
        x = self.act(x)
        x = self.outconv(x)
        return x

    def apply_weight_constraints(self):
        self.inconv.weight.data = torch.clamp(self.inconv.weight.data, min=0)
        self.outconv.weight.data = torch.clamp(self.outconv.weight.data, min=0)


class denoise_cnn_shifted_relu2(nn.Module):
    def __init__(self, n_in=1, n_out=1):
        super().__init__()

        alpha = 0

        self.inconv = nn.ConvTranspose2d(
            in_channels=n_in, out_channels=16, kernel_size=3, padding=1)
        self.act1 = nn.Softplus()
        self.inconv2 = nn.ConvTranspose2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.outconv = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.act2 = skip_act_sig(alpha, beta=1)
        self.outconv2 = nn.Conv2d(
            in_channels=16, out_channels=n_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.act1(x1)
        x2 = self.outconv(self.act(self.inconv2(x2)))
        x = x1+self.act2(x2, x1)
        x = self.outconv2(x)
        return x

    def apply_weight_constraints(self):
        self.inconv2.weight.data = torch.clamp(self.inconv2.weight.data, min=0)
        self.outconv.weight.data = torch.clamp(self.outconv.weight.data, min=0)


class denoise_cnn_soft_shrink(nn.Module):
    def __init__(self, n_in=1, n_out=1):
        super().__init__()

        alpha = 0.01

        self.inconv = nn.ConvTranspose2d(
            in_channels=n_in, out_channels=16, kernel_size=3, padding=1)
        self.act = soft_shrink(alpha=alpha)
        self.outconv = nn.Conv2d(
            in_channels=16, out_channels=n_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.inconv(x)
        x = self.act(x)
        x = self.outconv(x)
        return x


class denoise_cnn_soft_shrink_nobias(nn.Module):
    def __init__(self, n_in=1, n_out=1):
        super().__init__()

        alpha = 0.01

        self.inconv = nn.ConvTranspose2d(
            in_channels=n_in, out_channels=16, kernel_size=3, padding=1)
        self.act = soft_shrink_fix(alpha=alpha)
        self.outconv = nn.Conv2d(
            in_channels=16, out_channels=n_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.inconv(x)
        x = self.act(x)
        x = self.outconv(x)
        return x


class denoise_cnn_firm_shrink(nn.Module):
    def __init__(self, n_in=1, n_out=1):
        super().__init__()

        alpha = 0.01
        beta = 0.011

        self.inconv = nn.ConvTranspose2d(
            in_channels=n_in, out_channels=16, kernel_size=3, padding=1)
        self.act = firm_shrink(alpha=alpha, beta=beta)
        self.outconv = nn.Conv2d(
            in_channels=16, out_channels=n_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.inconv(x)
        x = self.act(x)
        x = self.outconv(x)
        return x
