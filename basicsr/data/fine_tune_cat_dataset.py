## 只有一帧短曝光和一帧长曝光进行引导

## 数据增强考虑：长曝光的亮度是随机n倍，由两张短曝光图像进行合成


import math
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import numpy as np
from PIL import Image, ImageEnhance

import glob
import random
import re
import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch
from basicsr.utils.registry import DATASET_REGISTRY
import os
import cv2
import random
def random_shake(tensor, i):
    """
    对 tensor 进行抖动，只有 i > 0 的时候才抖动
    包括平移和旋转
    """
    if i > 0:
        # 随机平移，模拟抖动
        max_shift = 5  # 最大平移的像素数
        shift_x = random.randint(0, max_shift)
        shift_y = random.randint(0, max_shift)

        # 随机旋转，模拟抖动
        max_rotation = 3  # 最大旋转角度（单位：度）
        rotation_angle = random.randint(0, max_rotation)

        # 创建一个变换，包含平移和旋转
        transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=rotation_angle,  # 随机旋转
                    translate=(
                        shift_x / tensor.size(1),
                        shift_y / tensor.size(2),
                    ),  # 随机平移
                )
            ]
        )

        # 对图像进行平移和旋转变换
        tensor = transform(tensor)

    return tensor

class Banding_Image_Loader(data.Dataset):
    def __init__(self, transform_base):
        self.real_input_list = []
        self.real_gt_list = []
        self.motion_path = []
        self.motion_path_list = []
        self.motion_gt_list = []
        
        self.img_size = transform_base["img_size"]

        self.transform_base = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomRotation([0, 90]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

        self.resize = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=(90, 90)),
                        transforms.RandomRotation(degrees=(180, 180)),
                        transforms.RandomRotation(degrees=(270, 270)),
                    ]
                ),
            ]
        )

    def __len__(self):
        return len(self.real_input_list)

    def __getitem__(self, index):

        choice = random.randint(0, 1)
        to_tensor = transforms.ToTensor()
        
        jiange = random.randint(1,3)
        
        # choice = 0
        
        if choice == 0:
            # 按需加载图像路径
            real_path = self.real_input_list[index]
            real_gt_path = self.real_gt_list[index]

            # 加载随机连续的三帧输入图像
            image_files = sorted(os.listdir(real_path))
            
            while len(image_files) <= jiange*2+1:
                jiange -= 1
            
            input_index = random.sample(range(0, len(image_files)-jiange*2-1), 1)
            
            input_index = [input_index[0], input_index[0] + jiange, input_index[0] + jiange*2]

            # 按需加载并转换为 Tensor

            # 示例：应用到你原来的代码中
            input_imgs = [
                random_shake(
                    to_tensor(
                        Image.open(
                            os.path.join(real_path, image_files[input_index[i]])
                        ).convert("RGB")
                    ),
                    i,
                )
                for i in range(0, 3)
            ]
            # 加载 GT 图像
            gt_img = to_tensor(
                Image.open(
                    os.path.join(real_gt_path, sorted(os.listdir(real_gt_path))[0])
                ).convert("RGB")
            )

            # 数据增
        else:
            index = index % len(self.motion_path_list)
            motion_path = self.motion_path_list[index]
            motion_gt_path = self.motion_gt_list[index]
            
            motion_files = sorted(os.listdir(motion_path))
            
            # print(len(motion_files)-jiange*2-1)
            
            while len(motion_files) <= jiange*2+1:
                jiange -= 1
            
            input_index = random.sample(range(0, len(motion_files)-jiange*2-1), 1)
            
            input_index = [input_index[0], input_index[0] + jiange, input_index[0] + jiange*2]
            
            input_imgs = [
                random_shake(
                    to_tensor(
                        Image.open(
                            os.path.join(motion_path, motion_files[input_index[i]])
                        ).convert("RGB")
                    ),
                    i,
                )
                for i in range(0, 3)
            ]
            
            gt_img = to_tensor(
                Image.open(
                    os.path.join(motion_gt_path, sorted(os.listdir(motion_gt_path))[input_index[0]])
                ).convert("RGB")
            )
            
        all_imgs = torch.cat(input_imgs + [gt_img], dim=0)
        all_imgs = self.transform_base(all_imgs)

        input0_img, input1_img, input2_img, gt_img = torch.split(all_imgs, 3, dim=0)
        
        input_img = torch.cat([input0_img, input1_img, input2_img], dim=0)
            
        return {"gt": gt_img, "input": input_img}

    def __len__(self):
        return len(self.real_input_list)

    def load_real_data(self, real_name, real_path):
        # self.ext = ["png", "jpeg", "jpg", "bmp", "JPG"]
        self.real_path = real_path
        # print(real_path, "1")
        self.real_input_list.extend(glob.glob(self.real_path + "/input/*/"))
        self.real_input_list.sort()
        # print(self.real_input_list)
        self.real_gt_list.extend(glob.glob(self.real_path + "/gt/*/"))
        self.real_gt_list.sort()

    def load_motion_data(self, motion_name, motion_path):
        self.ext = ["png", "jpeg", "jpg", "bmp", "tif"]
        self.motion_path = motion_path
        
        self.motion_path_list.extend(glob.glob(self.motion_path + "/input/*/"))
        self.motion_path_list.sort()
        
        self.motion_gt_list.extend(glob.glob(self.motion_path + "/gt/*/"))
        self.motion_gt_list.sort()
        

@DATASET_REGISTRY.register()
class Fine_tune_Loader_cat(Banding_Image_Loader):
    def __init__(self, opt):
        self.opt = opt
        Banding_Image_Loader.__init__(self, opt["transform_base"])

        real_dict = opt["input_path"]["real_path"]
        motion_dict = opt["input_path"]["motion_path"]

        self.load_real_data("real_path", real_dict)
        self.load_motion_data("motion_path", motion_dict)