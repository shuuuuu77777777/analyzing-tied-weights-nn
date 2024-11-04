import torch
import torch.nn as nn


def power_iteration2(operator, size, device, num_ite):
    vec = torch.ones(size).to(device)
    vec /= torch.norm(vec.view(size[0], -1),
                      dim=1, p=2).view(size[0], 1, 1, 1)

    with torch.no_grad():
        for i in range(num_ite):
            new_vec = operator(vec)
            new_vec = new_vec / torch.norm(new_vec.view(
                size[0], -1), dim=1, p=2).view(size[0], 1, 1, 1)
            old_vec = vec
            vec = new_vec

    new_vec = operator(vec)
    div = torch.norm(
        vec.view(size[0], -1), dim=1, p=2).view(size[0])
    eigenvalue = torch.abs(
        torch.sum(vec.view(size[0], -1) * new_vec.view(size[0], -1), dim=1)) / div
    return eigenvalue


def get_norm_jacobian(input, output, device):
    def operator(vec): return torch.autograd.grad(
        output, input, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]
    spector_norm_jaco = power_iteration2(
        operator, input.size(), device, num_ite=30)
    return spector_norm_jaco.max()


def get_norm_convlayer(layer_weight, image_size, device):
    weight_shape = layer_weight.shape
    conv_layer = nn.Conv2d(
        in_channels=weight_shape[1], out_channels=weight_shape[0], kernel_size=weight_shape[2], padding=weight_shape[2]//2, bias=False)
    convtrans_layer = nn.ConvTranspose2d(
        in_channels=weight_shape[0], out_channels=weight_shape[1], kernel_size=weight_shape[2], padding=weight_shape[2]//2, bias=False)
    conv_layer.weight.data = layer_weight
    convtrans_layer.weight.data = layer_weight

    input_size = [1, weight_shape[1], image_size, image_size]

    def operator(vec): return convtrans_layer(conv_layer(vec))
    norm_convlayer = power_iteration2(operator, input_size, device, num_ite=30)
    return norm_convlayer**0.5


def get_norm_convtlayer(layer_weight, image_size, device):
    weight_shape = layer_weight.shape
    convtrans_layer = nn.ConvTranspose2d(
        in_channels=weight_shape[1], out_channels=weight_shape[0], kernel_size=weight_shape[2], padding=weight_shape[2]//2, bias=False)
    conv_layer = nn.Conv2d(
        in_channels=weight_shape[0], out_channels=weight_shape[1], kernel_size=weight_shape[2], padding=weight_shape[2]//2, bias=False)
    conv_layer.weight.data = layer_weight
    convtrans_layer.weight.data = layer_weight

    input_size = [1, weight_shape[1], image_size, image_size]

    def operator(vec): return convtrans_layer(conv_layer(vec))
    norm_convlayer = power_iteration2(operator, input_size, device, num_ite=30)
    return norm_convlayer**0.5


def get_list_norm_convtrans(model, image_size, gray):
    convt_list = [
        module for name, module in model.named_modules() if isinstance(module, nn.ConvTranspose2d)]
    norm_list = torch.zeros([len(convt_list)])

    for i in range(len(convt_list)):
        convt_layer = convt_list[i]

        device = convt_layer.weight.device
        eigenvalue = get_norm_convtlayer(
            convt_layer.weight.data, image_size, device)

        norm_list[i] = eigenvalue

    return norm_list


def get_list_norm_conv(model, image_size, gray):
    conv_list = [
        module for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    norm_list = torch.zeros([len(conv_list)])

    for i in range(len(conv_list)):
        conv_layer = conv_list[i]

        device = conv_layer.weight.device

        eigenvalue = get_norm_convlayer(
            conv_layer.weight.data, image_size, device)

        norm_list[i] = eigenvalue

    return norm_list


def get_eig_H(kernel, kernel_size, image_size):
    kernel_row = kernel.flatten()
    root_row = torch.zeros(kernel_row.shape, dtype=torch.complex64)
    m = kernel_size
    n = image_size
    for i in range(m):
        root_row[i*m:(i+1)*m] = torch.exp(2*torch.pi*1j/n**2 *
                                          torch.arange(i*n, n*i+m))
    eig_row = torch.zeros(n**2, dtype=torch.complex64)
    for i in range(n**2):
        eig_row[i] = torch.sum(kernel_row*(root_row**(i+1)))
    return torch.abs(eig_row)**2


def get_sigma_and_tau(lip, eig_H, mu):

    kappa = torch.max(eig_H.abs()).item()-torch.min(eig_H.abs()).item()
    # kappa = torch.max(normH).item()
    rho = torch.min(eig_H.abs()).item()
    kappa = mu*kappa
    rho = mu*rho
    beta = 1/float(lip)

    sigma = rho*beta/(1-beta)
    tau = 0.8/(sigma+0.5*kappa)
    return sigma, tau, rho
