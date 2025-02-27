import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json
from warnings import filterwarnings
import shutil

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm
from os.path import join
import torch.nn.functional as F

# from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
import clip
from PIL import Image, ImageFile





# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    """axis=-1 表示默认计算 a 数组的最后一个轴上的范数（即行向量的L2范数，也可以理解为每个样本的范数），
    order=2 表示计算L2范数。因此，对于二维数组，函数将对每一行进行L2归一化。
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def pred(_path_to_data, _path_to_images, _image_filenames):
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    s = torch.load(
        "sac+logos+ava1-l14-linearMSE.pth"
    )  # load the model you trained previously or the model available in this repo

    model.load_state_dict(s)
    model.to("cuda")
    model.eval()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64

    image_score = []
    image_good = []
    for image in enumerate(tqdm.tqdm(_image_filenames)):
        img_path = _path_to_images + "/" + image[1]
        pil_image = Image.open(img_path)

        processed_image = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model2.encode_image(processed_image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())

        prediction = model(
            torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
        )

        score_np = prediction.data.cpu().numpy()
        score_np=np.squeeze(score_np)
        # print("Aesthetic score predicted by the model:")
        # print(prediction)
        image_score.append(dict(image_name=image[1], score=score_np))

    for image in image_score:
        if image.get("score") > 5.20:
            image_good.append(image)

    print(len(image_score))
    print(len(image_good))

    path_to_good=_path_to_data+'/good/'
    for image in enumerate(tqdm.tqdm(image_good)):
        image_name=image[1].get('image_name')
        image_path=_path_to_images+image_name
        shutil.copy(image_path,path_to_good+image_name)

    # 绘制直方图
    values = [d['score'] for d in image_score]
    plt.hist(values, bins=10)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Data')
    plt.show()


# 单独载入预测参数
def pre_pred():
    print('正在预处理')
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    s = torch.load(
        "C:/Users/123/Documents/GitHub/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"
    )  # load the model you trained previously or the model available in this repo

    model.load_state_dict(s)
    model.to("cuda")
    model.eval()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64
    return model, model2, preprocess


# 使用输入的模型进行预测
def to_pred(model, model2, preprocess, _path_to_data, _path_to_images, _img_name):
    img_path = _path_to_images + "/" + _img_name
    pil_image = Image.open(img_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model2.encode_image(processed_image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy())

    prediction = model(
        torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
    )

    score_np = prediction.data.cpu().numpy()
    score_np=np.squeeze(score_np)
    # print("Aesthetic score predicted by the model:")
    # print(prediction)
    return score_np


def pred_single(_path_to_data, _path_to_images, _img_name):
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    s = torch.load(
        "./sac+logos+ava1-l14-linearMSE.pth"
    )  # load the model you trained previously or the model available in this repo

    model.load_state_dict(s)
    model.to("cuda")
    model.eval()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64

    img_path = _path_to_images + "/" + _img_name
    pil_image = Image.open(img_path)

    processed_image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model2.encode_image(processed_image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy())

    prediction = model(
        torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
    )

    score_np = prediction.data.cpu().numpy()
    score_np=np.squeeze(score_np)
    # print("Aesthetic score predicted by the model:")
    print(score_np)

    path_to_good=_path_to_data+'/good/'
    image_name=_img_name
    image_path=_path_to_images+image_name
    shutil.copy(image_path,path_to_good+image_name)   


#####  This script will predict the aesthetic score for this image file:
img_name = "img_0000_145.jpg"
path_to_data = "D:/BaiduNetdiskDownload/group1/"

path_to_images = path_to_data + "/group1/"
image_filenames = []
for filename in os.listdir(path_to_images):
    if (
        filename.endswith(".jpg")
        or filename.endswith(".png")
        or filename.endswith(".jpeg")
    ):
        image_filenames.append(filename)



model, model2, proprecss=pre_pred()

# 实现持续运行
while True:
    print('please input\n')
    # 接受输入的参数
    input_param = input()
    print("params are: ", input_param)
    # 对参数进行处理
    if input_param == "exit":
        break
    score=to_pred(model, model2, proprecss, path_to_data, path_to_images,input_param)

    print(score)
    # 循环继续，等待下一个输入参数
    # pred(path_to_data, path_to_images,image_filenames)
        
    