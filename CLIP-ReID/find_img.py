from PIL import Image
import numpy as np
import os
import torch
from pytorch_msssim import ssim
import torchvision.transforms as transforms
from tqdm import tqdm

root = "/export/livia/home/vision/Rbhattacharya/work/data/data/msmt17/MSMT17/train"

def list_files(file_list, root):
    if root[-4:] in [".png", ".jpg"]: 
        file_list.append(root)
    elif ".DS_Store" in root: file_list = file_list
    else:
        for item in os.listdir(root):
            file_list = list_files(file_list=file_list, root=os.path.join(root, item))
        
    return file_list

all_files = list_files([], root)

def read_img(path, size=(512, 512)):
    img = Image.open(path)
    img = img.resize(size=size)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

sample_img_path = "/export/livia/home/vision/Rbhattacharya/work/data/data/msmt17/MSMT17/train/0327/0327_007_05_0113morning_0520_1_ex.jpg"
query = read_img(sample_img_path)
scores = []
for path in tqdm(all_files, total=len(all_files)):
    sample = read_img(path)
    mse = (sample - query) ** 2
    mse = mse.mean()
    s = ssim(query, sample, data_range=1, size_average=True)
    score = mse * 5 + (1 - s) * 2
    scores.append(score)

scores = np.array(scores)
indices = np.argsort(scores)

top_k = 5
print(f"Top {top_k} Predictions:")
for i in range(top_k):
    print(f"{1+i}: {all_files[indices[i]]}")