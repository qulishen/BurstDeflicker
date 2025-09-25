import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import math
import re
from torch.distributions import Normal
import torchvision.transforms as transforms
import os
from thop import profile
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage import io
from torchvision.transforms import ToTensor
import numpy as np
from glob import glob
import lpips

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--model_type", type=str, default="Uformer")
parser.add_argument(
    "--model_path", type=str, default="checkpoint/flare7kpp/net_g_last.pth"
)
parser.add_argument("--gt", type=str, default=None)
parser.add_argument("--mask", type=str, default=None)

def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor = ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    output_lpips = loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda())
    return output_lpips.cpu().detach().numpy()[0, 0, 0, 0]

def load_params(model_path):
    full_model = torch.load(model_path)
    if "params_ema" in full_model:
        return full_model["params_ema"]
    elif "params" in full_model:
        return full_model["params"]
    else:
        return full_model
    
def calculate_metrics(args):
    loss_fn_alex = lpips.LPIPS(net="alex").cuda()
    gt_folder = args["gt"]
    
    gt_list = []
    for subdir in os.listdir(gt_folder):
        subdir_path = os.path.join(gt_folder, subdir)
        if os.path.isdir(subdir_path):
            images = glob(os.path.join(subdir_path, "*"))  
            # print(images)
            if images:
                gt_list.append(images[0])
    
    gt_list = sorted(gt_list)
    input_list = []
    input_folder = result_path
    # print(gt_list)
    n = 0
    
    psnr, ssim, lpips_val = 0, 0, 0
    
    input_subdir = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    input_subdir = sorted(input_subdir)
    
    for subdir in tqdm(input_subdir, desc="Processing folders"):
        img_gt = io.imread(gt_list[n])
        ssim0, psnr0, lpips_val0 = 0, 0, 0

        num = 0
        sub_dir = os.listdir(os.path.join(input_folder, subdir))
        sub_dir = sorted(sub_dir)
        for file in sub_dir:
            img_input = io.imread(os.path.join(input_folder, subdir, file))
            ssim0 += compare_ssim(img_gt, img_input, multichannel=True)
            psnr0 += compare_psnr(img_gt, img_input, data_range=255)
            lpips_val0 += compare_lpips(img_gt, img_input, loss_fn_alex)
            num += 1
        ssim0 /= num
        psnr0 /= num
        lpips_val0 /= num
        ssim += ssim0
        psnr += psnr0
        lpips_val += lpips_val0
        n += 1
        print(f"n, PSNR: {psnr0}, SSIM: {ssim0}, LPIPS: {lpips_val0}")
    ## 所有组的平均
    psnr /= n
    ssim /= n
    lpips_val /= n
    print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_val}")
    
        
if __name__ == "__main__":
    args = parser.parse_args()

    images_path = os.path.join(args.input, "*.*")
    result_path = args.input

    eval_arg = vars(args)
    calculate_metrics(eval_arg)