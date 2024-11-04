import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def sig(beta, input):
    return 1/(1+torch.exp(-beta*input))


class sc_sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        result = sig(beta, x)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_y):
        result, = ctx.saved_tensors
        return grad_y*(1-result)*result, None


class sc_sigmoid_layer(nn.Module):

    def __init__(self, beta: int = 1) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, input) -> torch.Tensor:
        return sc_sigmoid.apply(input, self.beta)


class skip_act_sig(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.sc_layer = sc_sigmoid_layer(beta=beta)
        self.alpha = alpha

    def forward(self, x, x1):
        x1 = self.sc_layer(x1-self.alpha)
        x = x1*x
        return x


class multi_layer(nn.Module):

    def __init__(self, constant: int = 1) -> None:
        super().__init__()
        self.constant = constant

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input*self.constant


class ave_trans_layer(nn.Module):

    def __init__(self, constant: int = 1) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=constant),
            multi_layer(1/constant**2)
        )

    def forward(self, input) -> torch.Tensor:
        return self.layer(input)


class double_relu(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.layer(input)
        x = 0.5*x**2
        return x


class relu(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.layer(input)
        return x


class shifted_softplus(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.layer = nn.Softplus(beta=beta)
        self.alpha = alpha

    def forward(self, input):
        return self.layer(input-self.alpha)


class nonneg_conv(nn.Module):

    def __init__(self, in_channels, mid_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.layer = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, input):
        return self.layer(input)


class shifted_Relu(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.layer = nn.ReLU()
        self.alpha = alpha

    def forward(self, input):
        return self.layer(input-self.alpha)


class soft_shrink(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.layer = nn.ReLU()
        # self.alpha = alpha
        self.alpha = nn.Parameter(torch.tensor(
            alpha, dtype=torch.float32))

    def forward(self, input):
        input_m = -input
        output = self.layer(input-self.alpha)
        output_m = -self.layer(input_m-self.alpha)
        return output+output_m


class soft_shrink_fix(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.layer = nn.ReLU()
        self.alpha = alpha

    def forward(self, input):
        input_m = -input
        output = self.layer(input-self.alpha)
        output_m = -self.layer(input_m-self.alpha)
        return output+output_m


class firm_shrink(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.relu1 = soft_shrink_fix(alpha)
        self.relu2 = soft_shrink_fix(beta)

    def forward(self, input):
        output = (self.beta*self.relu1(input)-self.alpha *
                  self.relu2(input))/(self.beta-self.alpha)
        return output


class soft_shrink_reg_fix(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.layer = nn.Softplus(beta=beta)
        self.alpha = alpha
        self.beta = beta

    def forward(self, input):
        input_m = -input
        output = self.layer(input-self.alpha)
        output_m = -self.layer(input_m-self.alpha)
        return output+output_m


class skip_diff_soft_reg_fix(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.layer = sc_sigmoid_layer(beta=beta)

    def forward(self, input, skip_input):
        skip_input_m = -skip_input
        skip_output = self.layer(skip_input-self.alpha)
        skip_output_m = -self.layer(skip_input_m-self.alpha)
        skip = skip_output+skip_output_m
        return input*skip
