
from denoiser import denoise_grad2
from denoiser import denoise_grad_o
from denoiser import denoise_grad_o_for_2d

from denoiser import denoise_cnn_soft_shrink
from denoiser import denoise_cnn_soft_shrink_nobias
from denoiser import denoise_cnn_firm_shrink


def get_denoiser_and_para(net_name, image_channels, val):

    if net_name == "gradNet_onelayer":
        denoiser = denoise_grad_o(
            image_channels, image_channels)
        
    if net_name == "gradNet2":
        denoiser = denoise_grad2(
            image_channels, image_channels)

    if net_name == "soft_shrink":
        denoiser = denoise_cnn_soft_shrink(
            image_channels, image_channels)
    if net_name == "soft_shrink_nob":
        denoiser = denoise_cnn_soft_shrink_nobias(
            image_channels, image_channels)
    if net_name == "firm_shrink":
        denoiser = denoise_cnn_firm_shrink(
            image_channels, image_channels)
        
    if net_name == "2d":
        denoiser = denoise_grad_o_for_2d(
            image_channels, image_channels)
    return denoiser
