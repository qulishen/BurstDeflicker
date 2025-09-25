import torch
import torchvision
from PIL import Image

from basicsr.archs.Burstormer_arch import Burstormer
from basicsr.archs.HDRTransformer_arch import HDRTransformer
from basicsr.archs.Restormer_arch import Restormer
import argparse
import torchvision.transforms as transforms
import os
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
parser.add_argument(
    "--model_path", type=str, default="checkpoint/restormer.pth"
)
args = parser.parse_args()
images_path = os.path.join(args.input)
result_path = args.output
pretrain_dir = args.model_path

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def load_params(model_path):
    full_model = torch.load(model_path)
    if "params_ema" in full_model:
        return full_model["params_ema"]
    elif "params" in full_model:
        return full_model["params"]
    else:
        return full_model

def demo(images_path, output_path,  pretrain_dir):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(images_path)
    torch.cuda.empty_cache()

    model = (
        # Burstormer().cuda()
        Restormer().cuda()
        # HDRTransformer().cuda()
        # RetinexFormer()
    ) 

    model.load_state_dict(load_params(pretrain_dir))

    to_tensor = transforms.ToTensor()

    resize = transforms.Resize((512, 512))  # The output should in the shape of 128X

    for root, _, files in os.walk(images_path):
        files.sort()
        for file in files:
            if file.endswith(".png") or file.endswith(".JPG"):
            # Get current image index
                idx = files.index(file)

                # Read current image
                img1 = Image.open(os.path.join(root, file)).convert("RGB")
                img1_ori = to_tensor(img1).cuda()
                resize2org = transforms.Resize((img1.size[1], img1.size[0]))
                # img1 = img1.resize((img1.size[0] // 2, img1.size[1] // 2))
                # print(1)
                # img1 = torch.nn.functiona l.interpolate(img1, scale_factor=0.5)

                # Handle getting img2 and img3 based on position in sequence
                if idx + 2 < len(files):
                    # Normal case - get next 2 images
                    img2 = Image.open(os.path.join(root, files[idx + 1])).convert("RGB")
                    img3 = Image.open(os.path.join(root, files[idx + 2])).convert("RGB")
                elif idx + 1 < len(files):
                    # Only 1 image after current, get 1 before for img3
                    img2 = Image.open(os.path.join(root, files[idx + 1])).convert("RGB")
                    img3 = Image.open(os.path.join(root, files[idx - 1])).convert("RGB")
                else:
                    # No images after current, get 2 before
                    img2 = Image.open(os.path.join(root, files[idx - 2])).convert("RGB")
                    img3 = Image.open(os.path.join(root, files[idx - 1])).convert("RGB")

                img1 = resize(to_tensor(img1).cuda())
                img2 = resize(to_tensor(img2).cuda())
                img3 = resize(to_tensor(img3).cuda())

                model.eval()
                with torch.no_grad():

                    output_img = model(torch.cat([img1, img2, img3], dim=0).unsqueeze(0))
                    # output_img = model(torch.stack([img1, img2, img3], dim=0).unsqueeze(0))

                    output_img = img1_ori.unsqueeze(0) + resize2org(output_img - img1)

                    output_file = os.path.join(output_path, str(idx) + ".png")
                    torchvision.utils.save_image(output_img, output_file)

def get_subfolders(folder_path):
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    return subfolders

folders = get_subfolders(images_path)

for path in folders:
    demo(path, os.path.join(result_path, path.split("/")[-1]), pretrain_dir)
