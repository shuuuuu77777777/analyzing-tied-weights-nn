import torch
import torch.nn as nn


@torch.no_grad()
def weight_tie(model):
    conv_list = [
        module for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    convt_list = [
        module for name, module in model.named_modules() if isinstance(module, nn.ConvTranspose2d)]

    for conv_layer, convt_layer in zip(conv_list, reversed(convt_list)):
        conv_layer.weight.copy_(convt_layer.weight.clone().detach())


@torch.no_grad()
def requires_grad_False(model):
    if isinstance(model, nn.Conv2d):
        for param in model.parameters():
            param.requires_grad = False


def print_para(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Values: {param}")


def tie_process(model):
    # model.apply(enforch_positive_weight)
    weight_tie(model)
    # model.apply(requires_grad_False)


"""

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=2, padding=1, bias=False),
            nn.ConvTranspose2d(
                in_channels=1, out_channels=1, kernel_size=2, padding=1, bias=False)
        )
        self.conv3 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# 2. モデルのインスタンス化
model = MyModel()

print("none")
print_para(model)


model.apply(enforch_positive_weight)

print("abs")
print_para(model)

weight_tie(model)

print("tie")
print_para(model)

model.apply(requires_grad_False)

print("false")
print_para(model)



# 3. 重みの共有
# conv1の重みを取得
conv1_weight = model.conv1.weight
# conv2の重みを共有する
model.conv2.weight = nn.Parameter(conv1_weight.clone().detach())

conv2_weight = model.conv2.weight

print(conv1_weight)
print(conv2_weight)


for param in model.conv2.parameters():
    param.requires_grad = False

print(conv1_weight)
print(conv2_weight)

model.apply(enforch_positive_weight)
print(conv1_weight)
print(conv2_weight)
"""
