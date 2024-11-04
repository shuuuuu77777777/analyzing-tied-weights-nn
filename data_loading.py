import torch
# import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
# import math


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor, seed=[]):
        if seed == []:
            if self.std == "random":
                std = 0.2*torch.rand(1).item()
            else:
                std = self.std
            noise_img = tensor + \
                torch.randn(tensor.size()) * std + self.mean
        
        else:
            if self.std == "random":
                std = 0.2*torch.rand(1).item()
            else:
                std = self.std
            generator = torch.Generator().manual_seed(seed)
            noise_img = tensor + \
                torch.randn(tensor.size(), generator=generator) * \
                std + self.mean
        return torch.clamp(noise_img, 0, 1), std

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddBlur(object):
    def __init__(self, blur_layer, device):
        self.blur_layer = blur_layer
        self.device = device

    def __call__(self, tensor):
        self.blur_layer.eval()
        tensor = tensor.unsqueeze(0).to(self.device)
        blur_image = self.blur_layer(tensor)
        
        blur_image = blur_image.squeeze(0).cpu()
        return blur_image

    def __repr__(self):
        return self.__class__.__name__ + '(blur_kernel={0})'.format(self.blur_layer)


class noise_data_loading(Dataset):

    def __init__(self, img_dir, noise_lev, gray=True, crop_size=256):
        self.img_paths = self._get_img_paths(img_dir)

        image_size = (crop_size, crop_size)
        self.noise_lev = noise_lev

        if gray:
            self.transform = transforms.Compose([
                transforms.RandomCrop(size=image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(size=image_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        self.add_noise = transforms.Compose([
            AddGaussianNoise(0, self.noise_lev)
        ])

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        true_img = self.transform(img)
        noise_img = self.add_noise(true_img)
        return {"true": true_img, "noise": noise_img}

    def _get_img_paths(self, img_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in [".jpg"]
        ]

        return img_paths

    def __len__(self):
        return len(self.img_paths)


class data_loading(Dataset):

    def __init__(self, device, blur_layer, img_dir, noise_lev, gray=True, crop_size=256):
        self.img_paths = self._get_img_paths(img_dir)

        image_size = (crop_size, crop_size)
        self.noise_lev = noise_lev

        if gray:
            self.transform = transforms.Compose([
                transforms.RandomCrop(size=image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(size=image_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        self.add_noise = transforms.Compose([
            AddGaussianNoise(0, self.noise_lev)
        ])
        self.blur_image = transforms.Compose([
            AddBlur(blur_layer, device)
        ])

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        true_img = self.transform(img)
        blur_img = self.blur_image(true_img)
        noise_img, noise_lev = self.add_noise(blur_img)
        return {"true": true_img, "noise": noise_img, "blur": blur_img, "noise_lev": noise_lev}

    def _get_img_paths(self, img_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in [".jpg", ".png"]
        ]

        return img_paths

    def __len__(self):
        return len(self.img_paths)


class noise_data_loading_test(Dataset):

    def __init__(self, img_dir, noise_lev, gray=True, crop_size=256):
        self.img_paths = self._get_img_paths(img_dir)

        image_size = (crop_size, crop_size)
        self.noise_lev = noise_lev
        self.seed = torch.randint(
            0, int(len(self.img_paths)*100), [len(self.img_paths)])

        if gray:
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=image_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        self.add_noise = AddGaussianNoise(0, self.noise_lev)

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        true_img = self.transform(img)
        noise_img = self.add_noise(true_img, self.seed[index].item())
        return {"true": true_img, "noise": noise_img}

    def _get_img_paths(self, img_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in [".jpg"]
        ]

        return img_paths

    def __len__(self):
        return len(self.img_paths)


class data_loading_test(Dataset):

    def __init__(self, device, blur_layer, img_dir, noise_lev, gray=True, crop_size=256):
        self.img_paths = self._get_img_paths(img_dir)

        image_size = (crop_size, crop_size)
        if noise_lev == "random":
            self.noise_lev = 0.1
        else:
            self.noise_lev = noise_lev
        self.seed = torch.randint(
            0, int(len(self.img_paths)*100), [len(self.img_paths)])

        if gray:
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=image_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        self.add_noise = AddGaussianNoise(
            0, self.noise_lev)
        self.blur_image = transforms.Compose([
            AddBlur(blur_layer, device)
        ])

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        true_img = self.transform(img)

        ####################################
        blur_img = true_img

        # blur_img = self.blur_image(true_img)
        # print(blur_img.dtype)
        ####################################
        
        noise_img, _ = self.add_noise(blur_img, self.seed[index].item())
        return {"true": true_img, "noise": noise_img,  "blur": blur_img}

    def _get_img_paths(self, img_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in [".jpg"]
        ]

        return img_paths

    def __len__(self):
        return len(self.img_paths)


"""
train_image_dir = "../BSDS300/images/train"
test_image_dir = "../BSDS300/images/test"
bach_size = 13
noise_lev = 0.1

train_dataset = noise_data_loading(train_image_dir, noise_lev)
test_dataset = noise_data_loading_test(test_image_dir, noise_lev)
train_dataloader = DataLoader(train_dataset, bach_size)
test_dataloader = DataLoader(test_dataset, 2)

for i in test_dataloader:
    image_tensor = i["true"][0, :, :, :]
    noise_tensor = i["noise"][0, :, :, :]
    noise1 = image_tensor-noise_tensor

    # plt.imshow(noise_tensor.squeeze().numpy(), cmap='gray')
    # plt.axis('off')
    # plt.show()

for i in test_dataloader:
    image_tensor = i["true"][0, :, :, :]
    noise_tensor = i["noise"][0, :, :, :]
    noise2 = image_tensor-noise_tensor
print(torch.min(noise1-noise2))
print(torch.max(noise1-noise2))
"""
