import torch
import torch.nn as nn
import torch.nn.functional as F


from get_denoiser import get_denoiser_and_para
from get_para import get_sigma_and_tau
from get_para import get_eig_H
from get_para import get_list_norm_convtrans
from tie_weight import tie_process


class data_f_grad(nn.Module):
    def __init__(self,  kernel, crop_size, device):
        super().__init__()
        self.kernel = kernel
        self.H = H(self.kernel, crop_size, device)
        self.TransH = Ht(self.kernel, crop_size, device)
        self.mu = 1  # muの初期値
        if kernel.numel() > 0:
            self.kernel_size = kernel.shape[-1]
        else:
            self.kernel_size = 0

    def forward(self, x, y):
        #return torch.real(self.mu*self.TransH(self.H(x)-(y+0j)))
        return torch.real(self.mu*self.TransH(self.H(x)-y))


class PnP_Net(nn.Module):
    def __init__(self, net_name, kernel, crop_size, device, image_channels=3, do_fix=True, val=False):
        super().__init__()

        self.denoiser = get_denoiser_and_para(
            net_name, image_channels, val)

        self.grad = data_f_grad(kernel, crop_size, device)
        self.prox_f = prox_fid(kernel, crop_size, device)

        self.do_fix = do_fix
        self.net_name = net_name

        """
        self.param_list_convt = [
            param
            for name, module in self.denoiser.named_modules()
            if isinstance(module, nn.ConvTranspose2d)
            for param_name, param in module.named_parameters()
        ]
        self.param_list_conv = [
            param
            for name, module in self.denoiser.named_modules()
            if isinstance(module, nn.Conv2d)
            for param_name, param in module.named_parameters()
        ]
        """

        self.sigma = 1
        self.tau = 1
        self.rho = 1

    def forward_denoiser(self, x):
        return self.denoiser(x)

    # def forward(self, x, y):
    #     #return self.denoiser(x-self.grad(x, y))
    #     return self.denoiser(x-(x-y))
    
    def forward(self, x, y,sigma):
        #return self.denoiser(x-1.5*self.grad(x, y))
        return self.denoiser(x-sigma*(x-y))

    def forward_primal_dual(self, x, u, y):
        u_kari = u+self.sigma*x
        u_plus = u_kari-self.sigma * \
            self.denoiser((self.sigma+self.rho)**-1*u_kari)
        x = (x+self.tau*self.rho*x)-self.tau * \
            self.grad(x, y)-self.tau*(2*u_plus-u)
        return x, u_plus

    def forward_DRS(self, x, y_observed):
        y_iteration = self.prox_f(x, y_observed)
        z = self.denoiser(2*y_iteration-x)
        x = x + 0.8 * (z-y_iteration)  # beta=0.8
        return x

    def forward_HQS(self, x, y_observed, gamma=[]):
        z = self.prox_f(x, y_observed, gamma)
        x = self.denoiser(z)
        return x

    def forward_ADMM(self, x, v, u, y_observed, gamma=[]):
        x = self.prox_f(v-u, y_observed, gamma)
        v = self.denoiser(x+u)
        u = u+x-v
        return x, v, u

    def compute_convexity(self):
        return self.get_convexity(self)

    def tie_weight(self):
        if self.do_fix:
            tie_process(self.denoiser)

    def compute_sigma_and_tau(self, image_size, gray, mu=1, lip=[]):
        if lip == []:
            denoiser_liplist = get_list_norm_convtrans(
                self.denoiser, image_size, gray)
            denoiser_lip = self.denoiser.get_lip(denoiser_liplist)
        else:
            denoiser_lip = lip
        if self.grad.kernel_size == 0:
            self.sigma, self.tau, self.rho = get_sigma_and_tau(
                denoiser_lip, torch.tensor([1]), mu)
        else:
            eig_H = get_eig_H(self.grad.kernel,
                              self.grad.kernel_size, image_size)
            self.sigma, self.tau, self.rho = get_sigma_and_tau(
                denoiser_lip, eig_H, mu)
        return self.sigma, self.tau, self.rho

    def compute_sigma_to_para(self, image_size, gray, sigma_input, tau_input, denoiser_lip=[]):
        if denoiser_lip == []:
            denoiser_liplist = get_list_norm_convtrans(
                self.denoiser, image_size, gray)
            denoiser_lip = self.denoiser.get_lip(denoiser_liplist)
        if self.grad.kernel.numel() > 0:
            eig_H = get_eig_H(self.grad.kernel,
                              self.grad.kernel_size, image_size)
        else:
            eig_H = torch.tensor([1])
        sigma, tau, rho = get_sigma_and_tau(
            denoiser_lip, eig_H, 1)
        n = sigma_input/sigma
        rho = n*rho
        if tau_input == []:
            tau = tau/n
        else:
            tau = tau_input
        self.sigma = sigma_input
        self.rho = rho
        self.tau = tau
        return sigma_input, tau, rho


def get_eig_H(kernel, kernel_size, image_size):
    kernel_row = kernel.flatten()
    root_row = torch.zeros(kernel_row.shape, dtype=torch.complex64)
    m = kernel_size
    n = image_size
    for i in range(m):
        root_row[i*m:(i+1)*m] = torch.exp(2*torch.pi*1j/n**2 *
                                          (torch.arange(i*n, n*i+m)-((n*(kernel_size//2))+kernel_size//2)))
    eig_row = torch.zeros(n**2, dtype=torch.complex64)
    half_n = n**2//2
    for i in range(half_n):
        eig_row[i] = torch.sum(kernel_row*(root_row**(i)))
    eig_row[0] = torch.real(torch.sum(kernel_row))
    eig_row[half_n] = torch.real(torch.sum(kernel_row*(root_row**(half_n))))
    eig_row[half_n+1:] = torch.conj(torch.flip(eig_row[1:half_n], dims=[0]))
    return eig_row


def conv_layer(image_tensor, eig_row):
    image_tensor_size = image_tensor.size()
    batch_size = image_tensor_size[0]
    color_size = image_tensor_size[1]
    image_size = image_tensor_size[2]
    tensor_flat = image_tensor.reshape(
        batch_size, color_size, image_size**2).to(image_tensor.device)
    fft_tensor = torch.fft.fft(tensor_flat, dim=-1)

    complex_tensor = fft_tensor*eig_row.unsqueeze(0).unsqueeze(0)

    ifft_tensor = torch.fft.ifft(complex_tensor)

    result_tensor = ifft_tensor.view(
        batch_size, color_size, image_size, image_size)
    # result_tensor = torch.real(ifft_tensor.view(batch_size, color_size, image_size, image_size))
    # result_tensor = torch.abs((ifft_tensor.view(batch_size, color_size, image_size, image_size)))
    return result_tensor


def inv_conv_layer(image_tensor, eig_row):
    inv_eig_row = 1/eig_row
    print(torch.sum(torch.real(eig_row*inv_eig_row)))
    result_tensor = torch.real(conv_layer(image_tensor, inv_eig_row))
    return result_tensor


class H(nn.Module):
    def __init__(self, kernel, image_size, device):
        super().__init__()
        self.kernel = kernel
        if kernel.numel() > 0:
            self.eig_row = get_eig_H(
                kernel, kernel.shape[-1], image_size).to(device)
            self.image_size = image_size
            self.convlayer = Conv2d(self.eig_row)

    def forward(self, image):
        if self.kernel.numel() > 0:
            return self.convlayer(image)
        else:
            return image


class H_layer(nn.Module):
    def __init__(self, kernel, image_size, device):
        super().__init__()
        self.H = H(kernel, image_size, device)

    def forward(self, image):
        return torch.real(self.H(image))


class Ht(nn.Module):
    def __init__(self, kernel, image_size, device):
        super().__init__()
        self.kernel = kernel
        if kernel.numel() > 0:
            self.eig_row = get_eig_H(
                kernel, kernel.shape[-1], image_size).to(device)
            self.image_size = image_size
            self.convtlayer = Convtrans2d(self.eig_row)

    def forward(self, image):
        if self.kernel.numel() > 0:
            return self.convtlayer(image)
        else:
            return image


class Ht_layer(nn.Module):
    def __init__(self, kernel, image_size, device):
        super().__init__()
        self.Ht = Ht(kernel, image_size, device)

    def forward(self, image):
        return torch.real(self.Ht(image))


class Conv2d(nn.Module):
    def __init__(self, eig_row):
        super().__init__()
        self.eig_row = eig_row

    def forward(self, image):
        return conv_layer(image, self.eig_row)


class Convtrans2d(nn.Module):
    def __init__(self, eig_row):
        super().__init__()
        self.eig_row = torch.conj(eig_row)

    def forward(self, image):
        return conv_layer(image, self.eig_row)


class prox_fid(nn.Module):
    def __init__(self, kernel, image_size, device):
        super().__init__()

        # f=0.5||Hx-y||^2のprox
        # prox_{gamma*f}(x) = (I+gamma*HtH)^(-1)(gamma*Ht*y+x)
        self.kernel = kernel
        self.gamma = 0.8  # prox_{gamma*f}のgamma
        if kernel.numel() > 0:
            self.eig_row_H = get_eig_H(
                kernel, kernel.shape[-1], image_size).to(device)
            self.eig_row_inv = 1/(1+self.gamma *
                                  torch.abs(self.eig_row_H)**2).to(device)
            self.inv_layer = Conv2d(self.eig_row_inv)
            self.trans_layer = Convtrans2d(self.eig_row_H)

    def forward(self, x, y, gamma=[]):
        if gamma == []:
            gamma = self.gamma
        if self.kernel.numel() > 0:
            #return torch.real(self.inv_layer(gamma*self.trans_layer(y)+(x+0j)))
            return torch.real(self.inv_layer(gamma*self.trans_layer(y)+x))
        else:
            return (1+gamma)*(gamma*y+x)