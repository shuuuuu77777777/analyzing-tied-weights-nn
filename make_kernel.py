import torch


def make_gaussblurk(n, sigma):

    # sigma = 4  # 分散

    ax = torch.linspace(-(n//2), n//2, n)
    x, y = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(x**2+y**2)/(2*sigma**2))
    kernel = kernel/torch.sum(kernel)
    return kernel.unsqueeze(0).unsqueeze(0)


def make_identityk(n):
    kernel = torch.zeros([n, n])
    kernel[n//2, n//2] = 1
    return kernel.unsqueeze(0).unsqueeze(0)


def make_squarek(n):
    kernel = torch.ones([n, n])
    kernel = kernel/torch.sum(kernel)
    return kernel.unsqueeze(0).unsqueeze(0)
