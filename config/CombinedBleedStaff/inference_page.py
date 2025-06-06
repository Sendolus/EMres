"""
Script for performing inference full recto and verso pages simultaneously
"""
import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils
import shutil
import pickle
import cv2
import matplotlib.pyplot as plt

import numpy as np
import torch
from IPython import embed

import options as option
from models import create_model
from helper_functions import inference_page_rv, align_pages_by_hand

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr


# OPTIONS:
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default=r"options/test/ir-sde.yml", help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

# Setup neural network
model = create_model(opt)
device = model.device
sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)

# Load the metadata
parsed_pickle_name = r'datasets/alamire_res_dict_clean/Copenhagen_10Copenhagen_10.pickle'
with open(parsed_pickle_name, 'rb') as f:
    parsed_folium_dict = pickle.load(f)

# Load the images
original_image_path = r"datasets/Page_example/Copenhagen_10.jpeg"
recto_path = r"datasets/Page_example/recto.jpg"
verso_path = r"datasets/Page_example/verso.jpg"
rmask_path = r"datasets/Page_example/recto_staffs.jpg"
vmask_path = r"datasets/Page_example/verso_staffs.jpg"
rsmask_path = r"datasets/Page_example/recto_staff_lines.jpg"
vsmask_path = r"datasets/Page_example/verso_staff_lines.jpg"

original_image_img = cv2.imread(original_image_path)
recto_img = cv2.imread(img_path)
recto_img_rbg = cv2.cvtColor(recto_img, cv2.COLOR_BGR2RGB)
verso_img_aligned = cv2.imread(verso_path)
verso_img_rgb = cv2.cvtColor(verso_img_aligned, cv2.COLOR_BGR2RGB)
rmask_img = cv2.imread(rmask_path)
vmask_img = cv2.imread(vmask_path)
rsmask_img = cv2.imread(rsmask_path)
vsmask_img = cv2.imread(vsmask_path)


# Perform the restoration
print("-----starting inference-----")
resultR, resultV = inference_page_rv(parsed_folium_dict, recto_img, verso_img_aligned, rmask_img, vmask_img, rsmask_img, vsmask_img, original_image_img, model, sde, 0)


# Display the results
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create 1 row, 2 columns of subplots
axes[0].imshow(recto_img_rbg)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(resultR)
axes[1].set_title("Processed Image")
axes[1].axis("off")
plt.show()

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
axes2[0].imshow(cv2.flip(verso_img_rgb, 1))
axes2[0].set_title("Original Image 2")
axes2[0].axis("off")

axes2[1].imshow(cv2.flip(resultV, 1))
axes2[1].set_title("Processed Image 2")
axes2[1].axis("off")

plt.imsave("result_R.png", resultR)
plt.imsave("result_V.png", resultV)

plt.tight_layout()
plt.show()
