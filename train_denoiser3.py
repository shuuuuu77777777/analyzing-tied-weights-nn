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
from my_loss import mse_loss

from PnPnet import PnP_Net
from make_kernel import make_gaussblurk
from make_kernel import make_squarek
from tie_weight import tie_process


def get_kernel(kernel_name):
    if kernel_name == "gauss_blur":
        sigma = 1
        kernel_size = 13
        kernel = make_gaussblurk(kernel_size, sigma)
    if kernel_name == "gauss_blur1":
        sigma = 1.6
        kernel_size = 25
        kernel = make_gaussblurk(kernel_size, sigma)
    if kernel_name == "gauss_blur2":
        sigma = 3
        kernel_size = 25
        kernel = make_gaussblurk(kernel_size, sigma)
    if kernel_name == "square":
        kernel_size = 5
        kernel = make_squarek(kernel_size)
    if kernel_name == "noise_only":
        kernel_size = 1
        kernel = torch.tensor([])
    return kernel


def get_model(net_name, kernel, gray, weight_tie, crop_size, device):
    if gray:
        n_channels = 1
        n_out = 1
    else:
        n_channels = 3
        n_out = 3

    model = PnP_Net(net_name, kernel, crop_size, device,
                    n_channels, weight_tie)

    return model, weight_tie


def get_loss(net_name, batch_size, color, crop_size, device, special_list):
    loss = mse_loss()
    return loss


def get_para(net_name, weight_tie, special_list):

    kname = "noise_only"
    print(net_name)
    print(kname)

    net_name = net_name
    kernel_name = kname
    crop_size = 64
    #crop_size = 28

    noise_lev = 0.1

    gray = True  # gray画像か否か
    use_gpu = True  # gpu使うか

    weight_tie = weight_tie  # wegitを共有するか,absをするか
    use_wandb = True

    device = torch.device("cuda" if use_gpu else "cpu")

    kernel = get_kernel(kernel_name)
    model, weight_tie = get_model(
        net_name, kernel, gray, weight_tie, crop_size, device)

    use_google_colab = False

    wandb_project_name = "train_denoiser3" + (
        "google_colab" if use_google_colab else "")

    print(device.type)
    model.to(device=device)
    if use_google_colab:
        train_path = "/content/BSDS300/images/train"
        # train_path = "../Flickr2K"
        test_path = "/content/BSDS300/images/test"
    else:
        train_path = "/home/daito/analyzing-tied-weights-nn/BSDS300/images/train"
        # train_path = "../Flickr2K"
        test_path = "/home/daito/analyzing-tied-weights-nn/BSDS300/images/test"

    # if use_google_colab:
    #     train_path = "/content/BSDS300/images/train"
    #     # train_path = "../Flickr2K"
    #     test_path = "/content/BSDS300/images/test"
    # else:
    #     train_path = "C:\\Users\\toxic\\Desktop\\WGW\\aa\\MNIST Dataset JPG format\\MNIST - JPG - training\\0"
    #     # train_path = "../Flickr2K"
    #     test_path = "C:\\Users\\toxic\\Desktop\\WGW\\aa\\MNIST Dataset JPG format\\MNIST - JPG - testing\\0"

    epoch_size = 400
    batch_size: int = 10
    learning_rate: float = 5e-4
    weight_decay: float = 5e-7
    patience: int = 5
    grad_clip_th = 1e-2

    save_checkpoint = True

    checkpoint_state = {
        "use": False,
        "epoch": 400,
        "num": 7
    }

    if use_google_colab:
        checkpoint_path = "/content/drive/MyDrive/experiment/para_data/" + \
            wandb_project_name+"/"+net_name
        checkpoint_path_debug = "/content/drive/MyDrive/experiment/para_debug/" + \
            wandb_project_name+"/"+net_name
    else:
        checkpoint_path = "./para_data/"+wandb_project_name+"/"+net_name
        checkpoint_path_debug = "./para_debug/"+wandb_project_name+"/"+net_name
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(checkpoint_path_debug, exist_ok=True)
    if not os.path.isfile(checkpoint_path_debug+"/num.txt"):
        with open(checkpoint_path_debug+"/num.txt", "w") as file:
            file.write("1")
    if os.path.isfile(checkpoint_path_debug+"/num.txt"):
        with open(checkpoint_path_debug+"/num.txt", "r") as file:
            content = file.read().strip()

    wandb_run_name = net_name+"_noise=" + \
        str(noise_lev)+"_"+str("tie" if weight_tie else "notie")

    text_to_write = f"""\
    net_name: {net_name}
    noise_lev: {noise_lev}
    crop_size: {crop_size}
    gray: {"gray" if gray else "color"}
    next_num : {int(content)}
    """
    if save_checkpoint:
        with open(checkpoint_path + "/state.txt", "w") as logg:
            logg.write(text_to_write)

    if checkpoint_state["use"]:
        model_state = torch.load(checkpoint_path_debug+"/"+str(checkpoint_state["num"])+"_"+wandb_run_name + "_epoch_"+str(checkpoint_state["epoch"]) +
                                ".pth", map_location=device)
        model.denoiser.load_state_dict(model_state)

    return model, device, train_path, test_path, noise_lev, gray, crop_size, epoch_size, batch_size, learning_rate, weight_decay, patience, grad_clip_th, use_wandb, wandb_run_name, wandb_project_name, checkpoint_path, checkpoint_path_debug, content, checkpoint_state, save_checkpoint, net_name


def train_model(
        model,
        device,
        train_path,
        test_path,
        noise_lev,
        gray,
        crop_size,
        checkpoint_path,
        checkpoint_path_debug,
        content,
        checkpoint_state,
        save_checkpoint,
        net_name,
        special_list,
        loss_function,
        epoch_size: int = 5,
        batch_size: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 3,
        grad_clip_th: float = 1e-2,
        use_wandb: bool = False,
        wandb_run_name=None,
        wandb_project_name=None
):
    time1 = time.time()

    train_dataset = data_loading(
        device, model.grad.H, train_path, noise_lev, gray, crop_size)
    test_dataset = data_loading_test(device, model.grad.H,
                                     test_path, noise_lev, gray, crop_size)


    # train_dataset = noise_data_loading(
    #     train_path, noise_lev, gray, crop_size)
    # test_dataset = noise_data_loading_test(
    #                                  test_path, noise_lev, gray, crop_size)
    

    train_dataloader = DataLoader(train_dataset, batch_size)
    test_dataloader = DataLoader(test_dataset, 1)

    



    # wandbのlogを作成
    if use_wandb:
        experiment = wandb.init(
            project=wandb_project_name, name=content+"_"+wandb_run_name, settings=wandb.Settings(save_code=False))
        experiment.config.update(dict(epochs=epoch_size,
                                      noise_lev=noise_lev,
                                      batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      save_checkpoint=True)
                                 )
    else:
        experiment = []

    optimizer = optim.Adam(
        model.denoiser.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.Adam(model.param_list, lr=learning_rate, weight_decay=weight_decay)
    lr_th = 1e-5

    # もしメモリが問題ならforeachをFalseに
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=patience, factor=0.5)
    grad_scaler = torch.cuda.amp.GradScaler(
        enabled=True if device.type != "cpu" else False)
    criterion = loss_function

    # 確認用の変数
    global_step = 0
    step = 0

    val_check_interval = 1

    # loss_array = np.zeros([epoch_size, math.ceil(len(train_dataset)/batch_size)])
    # eva_array = np.zeros([epoch_size, math.ceil(len(train_dataset)/batch_size)//3])

    model.tie_weight()

    epoch_th = 10

    # min_val_score = 1
    # min_val_ind = 0
    # step_th = 100

    lr_th_reduce_weight = learning_rate/5

    # training
    for epoch in range(1, epoch_size+1):
        start_time = time.time()
        model.train()

        batch_step = 0
        # バーの表示
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epoch_size}', unit='img') as pbar:

            # データセットのとりだし
            for batch in train_dataloader:
                noise_image, true_image = batch["noise"],  batch["true"]
                noise_image = torch.tensor(noise_image)
                # true_image  = torch.tensor(true_image)
                
                noise_image = noise_image.to(
                    device=device, dtype=torch.float32).requires_grad_(True)
                true_image = true_image.to(
                    device=device, dtype=torch.float32).requires_grad_(True)

                # 学習の方法

                restoration_image = model.denoiser(noise_image)

                # torch.autograd.set_detect_anomaly(True)

                loss = criterion(restoration_image,
                                true_image)

                # トレーニング
                # print(f"Loss before backward: {loss.item()}")
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.denoiser.parameters(), grad_clip_th)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(noise_image.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # lossの記録
                # if loss_array.shape[1] < batch_step:
                # logger.warning("err1")
                # else:
                # loss_array[epoch-1, batch_step] = loss.item()

                # ステップ数の変更
                batch_step += 1
                global_step += 1
                step += 1

                # モデルのweight_tie

                tie_process(model.denoiser)

                # ステップサイズの決定
                # min_neg_layer = model.compute_convexity()

                if use_wandb:
                    experiment.log({
                        # "train loss": loss.item() if loss.item() < 0.05 else 0.05,
                        "train loss": round(loss.item(), 6),
                        "epoch": epoch,
                        "train_noiselev": (torch.sum(batch["noise_lev"])/batch_size).item()
                    })

                # evaluate

                if (batch_step % math.ceil(math.ceil(len(train_dataset)/batch_size)*val_check_interval)) == 0:
              
                    # 評価
                    val_score, val_noise, _, _, _ = evaluate(
                        model, test_dataloader, device, global_step, len(test_dataset), special_list, net_name)

                    scheduler.step(val_score)
                    # scheduler.step(val_score)

                    # if eva_array.shape[1] < batch_step//3:
                    # logger.warning("err2")
                    # else:
                    # eva_array[epoch-1, batch_step//3-1] = val_score

                    """
                    if min_val_score > val_score:
                        min_val_score = val_score
                        min_val_ind = global_step
                    if min_val_score < 1 and (min_val_ind+step_th) < global_step:
                        do_nonneg = True
                        criterion.sum_w = 0
                        criterion.mse_w = 1
                        flag = True
                    """

                    # evaの記録
                    if use_wandb:
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'val_score': round(val_score.item(), 4) if val_score.item() < 1 else 1,
                            # 'noise_images': wandb.Image(noise_image_eva[0].cpu()),
                            # 'true_image': wandb.Image(true_image_eva[0].cpu()),
                            # 'restoration_image': wandb.Image(restoration_image_eva[0].cpu()),
                            # "restoration_image_train": wandb.Image(restoration_image[0].cpu()),
                            'noise_score': val_noise
                        })

            currnt_lr = optimizer.param_groups[0]["lr"]
            #if net_name not in special_list["neg"]+special_list["nonneg"]:
                # if currnt_lr < lr_th:
                #     print("learning rate has reached below th")
                #     break

            if currnt_lr < lr_th:
                print("learning rate has reached below th")
                break



        end_time = time.time()
        time_deff = round((end_time-start_time)/60)
        print(epoch, "epoch", time_deff//60, "hour", time_deff % 60, "min")

        if (epoch % epoch_th) == 0 and not epoch == 0:
            if epoch == epoch_th:
                with open(checkpoint_path_debug+"/num.txt", "w") as file:
                    next_num = int(content)+1
                    file.write(str(next_num))
                text_to_write = f"""\
net_name: {net_name}
noise_lev: {noise_lev}
crop_size: {crop_size}
gray: {"gray" if gray else "color"}
epoch_size: {epoch_size}
batch_size: {batch_size}
learning_rate: {learning_rate}
weight_decay: {weight_decay}
patience: {patience}
"""
                with open(checkpoint_path_debug + "/"+content+"_state.txt", "a") as logg:
                    logg.write(text_to_write)
                if checkpoint_state["use"]:
                    text_to_write = f"""\
num: {checkpoint_state["num"]}
epoch: {checkpoint_state["epoch"]}
"""
                    with open(checkpoint_path_debug + "/"+content+"_state.txt", "a") as logg:
                        logg.write(text_to_write)

            state_dict = model.denoiser.state_dict()
            torch.save(state_dict, str(checkpoint_path_debug +
                                       "/"+content+"_"+wandb_run_name+"_epoch_"+str(epoch)+".pth"))
            with open(checkpoint_path_debug + "/"+content+"_state.txt", "a") as logg:
                time3 = time.time()
                logg.write(f"{epoch}_time: " +
                           str(((time3-time1)//6)/10)+"min\n")

    if epoch <= epoch_th:
        with open(checkpoint_path_debug+"/num.txt", "w") as file:
            next_num = int(content)+1
            file.write(str(next_num))
        text_to_write = f"""\
net_name: {net_name}
noise_lev: {noise_lev}
crop_size: {crop_size}
gray: {"gray" if gray else "color"}
epoch_size: {epoch_size}
batch_size: {batch_size}
learning_rate: {learning_rate}
weight_decay: {weight_decay}
patience: {patience}
"""
        with open(checkpoint_path_debug + "/"+content+"_state.txt", "a") as logg:
            logg.write(text_to_write)

    time2 = time.time()
    print((time2-time1)//60, "min")
    with open(checkpoint_path_debug + "/"+content+"_state.txt", "a") as logg:
        logg.write(f"train_time: {((time2-time1)//6)/10} min\n")

    state_dict = model.denoiser.state_dict()
    torch.save(state_dict, str(checkpoint_path_debug +
                               "/"+content+"_"+wandb_run_name+"_end_epoch_"+str(epoch)+".pth"))
    print("save checkpoint_debug")

    if save_checkpoint:
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        state_dict = model.denoiser.state_dict()
        torch.save(state_dict, str(checkpoint_path +
                                   "/"+wandb_run_name+".pth"))
        # np.save(str(checkpoint_path+"/"+wandb_run_name +"_"+"evaluate_score"), eva_array)
        # np.save(str(checkpoint_path+"/"+wandb_run_name+"_"+"loss"), loss_array)
        text_to_write = f"""\
    net_name: {net_name}
    noise_lev: {noise_lev}
    crop_size: {crop_size}
    gray: {"gray" if gray else "color"}
    num : {content}
    """
        with open(checkpoint_path + "/state.txt", "w") as logg:
            logg.write(text_to_write)
        print("save checkpoint")

    experiment.finish()





def evaluate(model, dataloader, device, global_step, len_set, special_list, net_name):
    model.eval()
    score = 0
    score_noise = 0

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Validation round', unit="batch", leave=False):

        noise_image, true_image = batch["noise"], batch["true"]
        noise_image = noise_image.to(
            device=device, dtype=torch.float32).requires_grad_(True)
        true_image = true_image.to(device=device, dtype=torch.float32)

        
        



        with torch.no_grad():


            #if net_name in special_list["neg"]+special_list["nonneg"] + special_list["nonexp"]:
            #model.denoiser.val = True

            denoise_image = model(noise_image, noise_image)

            #if net_name in special_list["neg"]+special_list["nonneg"] + special_list["nonexp"]:
            #model.denoiser.val = False

        score_kari = F.mse_loss(true_image, denoise_image)
        score_noise_kari = F.mse_loss(true_image, noise_image)

        score += score_kari
        score_noise += score_noise_kari

        if i == (global_step % len_set):
            denoise_image_return = denoise_image
            true_image_return = true_image
            noise_image_return = noise_image

    model.train()
    return score, score_noise, denoise_image_return, true_image_return, noise_image_return


name_list = [
    "gradNet_onelayer",
    # "gradNet2",
    # "soft_shrink",
    # "soft_shrink_nob",
    # "firm_shrink"
]


for j in range(1):
    for i in range(len(name_list)):

        net_name = name_list[i]

        weight_tie = True
        model, device, train_path, test_path, noise_lev, gray, crop_size, epoch_size, batch_size, learning_rate, weight_decay, patience, grad_clip_th, use_wandb, wandb_run_name, wandb_project_name, checkpoint_path, checkpoint_path_debug, content, checkpoint_state, save_checkpoint, net_name = get_para(
            net_name, weight_tie, [])
        loss = get_loss(net_name, batch_size,
                        1 if gray else 3, crop_size, device, [])

        train_model(
            model,
            device,
            train_path,
            test_path,
            noise_lev,
            gray,
            crop_size,
            checkpoint_path,
            checkpoint_path_debug,
            content,
            checkpoint_state,
            save_checkpoint,
            net_name,
            [],
            loss,
            epoch_size,
            batch_size,
            learning_rate,
            weight_decay,
            patience,
            grad_clip_th,
            use_wandb,
            wandb_run_name,
            wandb_project_name
        )
