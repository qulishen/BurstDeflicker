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


def cosine_wave(p, m, phi,interval=0.2):
    """生成余弦波形，使用 PyTorch 实现，幅值在 0 和 1 之间。"""
    p /= 3
    return 1.41 * torch.cos(2.0 * math.pi * p * m + phi)


def square_wave(p, m, phi):
    """生成方波，幅值在 0 和 1 之间"""
    return 0.5 * (1 + np.sign(np.sin(2.0 * math.pi * p * m + phi)))


def triangle_wave(p, m, phi):
    """生成三角波，幅值在 0 和 1 之间"""
    return 2.0 * np.abs(np.mod((p * m + phi / (2.0 * math.pi)), 1.0) - 0.5)


def sawtooth_wave(p, m, phi):
    """生成锯齿波，幅值在 0 和 1 之间"""
    return np.mod((p * m + phi / (2.0 * math.pi)), 1.0)


# def half_wave_rectified_sine_wave(p, m, phi):
#     """生成半波整流的正弦波，使用 PyTorch 实现，幅值在 0 和 1 之间。"""
#     sine_wave = torch.cos(2.0 * math.pi * p * m + phi)
#     return torch.maximum(sine_wave, torch.tensor(0.0)) / 0.5


def half_wave_rectified_sine_wave_with_interval(p, m, phi, interval=0.2):
    """生成带有间隔的半波整流的正弦波，幅值在 0 和 1 之间"""
    # 假设T为半波中大于0部分的长度，这样interval就是半波的长度
    start = -phi / (2.0 * math.pi * p)
    T = 0.5 / p
    INTERVAL = T * interval

    result = np.zeros_like(m)
    # print(T + INTERVAL)
    # print(start)
    # print(start + T)
    for i, x in enumerate(m):
        times = math.ceil((x - start) / (T + INTERVAL))
        new_x = x - times * (T + INTERVAL)
        if new_x < start:
            new_x += T + INTERVAL
        if new_x < start + T:
            result[i] = np.sin(2.0 * math.pi * p * new_x + phi) / 0.5
        else:
            result[i] = 0

    return torch.tensor(result)


def full_wave_rectified_sine_wave(p, m, phi):
    """生成全波整流的正弦波，幅值在 0 和 1 之间"""
    sine_wave = np.sin(2.0 * math.pi * p * m + phi)
    return np.abs(sine_wave)


def pwm_wave(p, m, phi,interval=0.2):
    """
    生成 PWM 波形，使用 PyTorch 实现，幅值在 0 和 1 之间。
    duty_cycle: 占空比，默认值为 0.7，表示高电平持续时间占总周期的 70%。
    """
    duty_cycle = 1 - interval
    t = (p * m + phi / (2.0 * math.pi)) % 1.0
    return (
        torch.where(t < duty_cycle, torch.tensor(1.0), torch.tensor(0.0)) / duty_cycle
    )


# 将所有波形函数放入一个列表
# 将所有波形函数放入一个列表
# wave_functions = [
#     cosine_wave,
#     # square_wave,
#     # triangle_wave,
#     # sawtooth_wave,
#     # half_wave_rectified_sine_wave,
#     half_wave_rectified_sine_wave_with_interval,
#     # full_wave_rectified_sine_wave,
#     pwm_wave,
# ]
wave_functions = [
    (cosine_wave, 1),
    (half_wave_rectified_sine_wave_with_interval, 2),
    (pwm_wave, 3),
]
# device = torch.device("cuda")


def make_matrix_2D(p, height, width, phi, average_brightness):
    hang = random.randint(0, 1)
    wave_func, func_index = random.choice(wave_functions)
    
    k = random.uniform(0, average_brightness)

    interval = random.uniform(0.2, 0.4)
    
    m = torch.linspace(0, height - 1, height)
    p /= math.ceil(height / 512)
    flk_1d = torch.abs(wave_func(p, m, phi,interval))
    flk_2d_1 = flk_1d.unsqueeze(1).repeat(1, width)
    phi += torch.rand(1).item() * math.pi
    flk_1d = torch.abs(wave_func(p, m, phi,interval))
    flk_2d_2 = flk_1d.unsqueeze(1).repeat(1, width)
    phi += torch.rand(1).item() * math.pi
    flk_1d = torch.abs(wave_func(p, m, phi,interval))
    flk_2d_3 = flk_1d.unsqueeze(1).repeat(1, width)

    return (
        flk_2d_1.unsqueeze(0),
        flk_2d_2.unsqueeze(0),
        flk_2d_3.unsqueeze(0),
        k,
    )


def add_flicker(image, average_brightness):
    # k = random.uniform(0, 5)
    phi = torch.rand(1).item() * 2 * math.pi
    f_row = random.randint(15000, 30000)

    f_enf = random.uniform(50, 60)
    p = f_enf / f_row
    _, height, width = image.shape

    e1, e2, e3, k = make_matrix_2D(p, height, width, phi, average_brightness)

    # e11 = e1.clone()
    # for i in range(0, e1.shape[2], 3):
    # target_options = [(110, 130, 150), (255, 255, 255)]
    # target = random.choice(target_options)
    
    coef1 = 1 + ((torch.abs(e1) - 1) / (1 + k))
    
    # coef1 = torch.abs(e11) / 3
    coef2 = 1 + ((torch.abs(e2) - 1) / (1 + k))

    coef3 = 1 + ((torch.abs(e3) - 1) / (1 + k))
    
    image1, image2, image3 = torch.split(image, 3, dim=0)
    
    coef1 = torch.where(coef1 > 1, torch.tensor(1.0), coef1)
    coef2 = torch.where(coef2 > 1, torch.tensor(1.0), coef2)
    coef3 = torch.where(coef3 > 1, torch.tensor(1.0), coef3)
    
    image11 = coef1 * image1
    image2 = coef2 * image2
    image3 = coef3 * image3

    # torchvision.utils.save_image(coef1, "mask.png")

    image = torch.cat([image1, image11, image2, image3], dim=0)

    return image


def synthesize(image, random_seed=42):
    phi = torch.rand(1).item() * math.pi
    L = random.uniform(1, 3)
    f_row = random.randint(7000, 10000)  ## 如果是cos的花，f_row要扩大三倍，原因未知。
    l1 = random.uniform(1, 3) / L
    l2 = random.uniform(1, 6 - l1) / L
    l3 = (6 - l1 - l2) / L

    return add_flicker(image, phi, f_row, l1, l2, l3)


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


# # ===== 计算光流，获取运动掩码 =====
# def compute_optical_flow(gray_short, gray_long, threshold=1.0):
#     # 转换为灰度图像
#     # gray_short = cv2.cvtColor(short_img, cv2.COLOR_BGR2GRAY)
#     # gray_long = cv2.cvtColor(long_img, cv2.COLOR_BGR2GRAY)

#     # 计算光流（使用 Farneback 法）
#     flow = cv2.calcOpticalFlowFarneback(
#         gray_short, gray_long, None, 0.5, 3, 15, 3, 5, 1.2, 0
#     )

#     # 计算光流幅度
#     magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])


#     # 生成运动掩码
#     motion_mask = (magnitude > threshold).astype(np.uint8) * 255
class Banding_Image_Loader(data.Dataset):
    def __init__(self, transform_base):
        self.real_input_list = []
        self.real_gt_list = []
        self.synthetic_path = []
        self.synthetic_path_list = []
        self.img_size = transform_base["img_size"]

        self.transform_base = transforms.Compose(
            [
                # 以50%的概率随机旋转90度
                transforms.RandomRotation([0, 90]),
                # transforms.RandomCrop(
                #     (self.img_size, self.img_size),
                #     pad_if_needed=True,
                #     padding_mode="reflect",
                # ),
                # transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

    def __len__(self):
        return len(self.real_input_list)

    def __getitem__(self, index):

        to_tensor = transforms.ToTensor()
        
        # if os.path.dirname(self.synthetic_path_list[index]) != os.path.dirname(
        #     self.synthetic_path_list[index + 2]
        # ):
        #     index += 2
        # input_path = self.synthetic_path_list[index]

        input_path0 = self.synthetic_path_list[index]
        # print(input_path0)
        input_path1 = self.synthetic_path_list[index]
        #
        input_path2 = self.synthetic_path_list[index]
        
        input0_img = Image.open(input_path0).convert("RGB")
        input1_img = Image.open(input_path1).convert("RGB")
        input2_img = Image.open(input_path2).convert("RGB")

        input0_img = to_tensor(input0_img)
        average_brightness = input0_img.mean().item()
        input1_img = to_tensor(input1_img)
        input2_img = to_tensor(input2_img)
        
        input1_img = random_shake(input1_img, 1)
        input2_img = random_shake(input2_img, 1)
        
        resize = transforms.Resize((512, 512))
        
        input_all = torch.cat([input0_img, input1_img, input2_img], dim=0)

        input_all = resize(input_all)
        
        input_all = add_flicker(input_all,average_brightness)
        
        input_all = torch.clamp(input_all, min=0, max=1)

        input_all = self.transform_base(input_all)

        # input_all = self.transform_base(input_all)

        ori_img, input0_img, input1_img, input2_img = torch.split(
            input_all, 3, dim=0
        )

        # noise1=0.01*np.random.chisquare(df=1)
        # noise2=0.01*np.random.chisquare(df=1)
        # noise3=0.01*np.random.chisquare(df=1)
        
        # input0_img = Normal(input0_img, noise1).sample()
        # input1_img = Normal(input1_img, noise2).sample()
        # input2_img = Normal(input2_img, noise3).sample()
        
        input_img = torch.stack((input0_img, input1_img, input2_img), dim=0)
        
        # print(input_img.shape)
        # print(mask0_img.shape)
        # print(input_path, input_index)
        return {"gt": ori_img, "input": input_img}

    def __len__(self):
        return len(self.synthetic_path_list)

    def load_real_data(self, real_name, real_path):
        # self.ext = ["png", "jpeg", "jpg", "bmp", "JPG"]
        self.real_path = real_path
        # print(real_path, "1")
        self.real_input_list.extend(glob.glob(self.real_path + "/flicker/*/"))
        self.real_input_list.sort()
        # print(self.real_input_list)
        self.real_gt_list.extend(glob.glob(self.real_path + "/gt/*/"))
        self.real_gt_list.sort()

    def load_synthetic_data(self, synthetic_name, synthetic_path):
        self.ext = ["png", "jpeg", "jpg", "bmp", "tif"]
        self.synthetic_path = synthetic_path
        self.synthetic_path_list = []

        [
            self.synthetic_path_list.extend(
                glob.glob(self.synthetic_path + "/*/*." + e)
            )
            for e in self.ext
        ]
        self.synthetic_path_list.sort()


@DATASET_REGISTRY.register()
class Remove_Synthesis_Pair_Loader(Banding_Image_Loader):
    def __init__(self, opt):
        self.opt = opt
        Banding_Image_Loader.__init__(self, opt["transform_base"])

        # synthetic_dict = opt["synthetic_dict"]

        # real_dict = opt["input_path"]["real_path"]
        sys_dict = opt["input_path"]["sys_path"]

        # if "data_ratio" not in opt or len(opt["data_ratio"]) == 0:
        #     self.data_ratio = [1] * len(synthetic_dict)
        # else:
        #     self.data_ratio = opt["data_ratio"]

        # if len(real_dict) != 0:
        #     for key in real_dict.keys():
        #         self.load_real_data(key, real_dict[key])

        # self.load_real_data("real_path", real_dict)
        self.load_synthetic_data("sys_path", sys_dict)

        # banding_dict=opt['banding_dict']
        # gt_dict=opt['gt_dict']
        # if len(banding_dict) !=0:
        #     for key in banding_dict.keys():
        #         self.load_banding_image(key,banding_dict[key])

        # if len(gt_dict) !=0:
        #     for key in gt_dict.keys():
        #         self.load_gt_image(key,gt_dict[key])
if __name__ == "__main__":
    to_tensor = transforms.ToTensor()

    image1 = Image.open(
        "1.JPG"
    ).convert("RGB")
    image2 = Image.open(
        "1.JPG"
    ).convert("RGB")
    image3 = Image.open(
        "1.JPG"
    ).convert("RGB")

    
    resize = transforms.Resize((512, 512))

    image1 = to_tensor(image1)
    image2 = to_tensor(image2)
    image3 = to_tensor(image3)

    # Calculate and print the average brightness of image1
    average_brightness = image1.mean().item()
    print(f"Average brightness of image1: {average_brightness}")
    
    input_all = torch.cat([image1, image2, image3], dim=0)
    
    # blur_transform = transforms.GaussianBlur(21, sigma=(0.1, 3.0))
    # color_jitter = transforms.ColorJitter(brightness=(0.8, 3), hue=0.0)
    
    
    input_all = resize(input_all)
    input_all = add_flicker(input_all, average_brightness)
    # flare_DC_offset = np.random.uniform(-0.02, 0.02)
    # input_all = transform_base(input_all)

    ori_img, input0_img, input1_img, input2_img = torch.split(input_all, 3, dim=0)

    # input0_img = torch.clamp(input0_img + flare_DC_offset, min=0, max=1)

   
    torchvision.utils.save_image(input0_img, "temp.png")

    torchvision.utils.save_image(input1_img, "temp1.png")
    torchvision.utils.save_image(input2_img, "temp2.png")