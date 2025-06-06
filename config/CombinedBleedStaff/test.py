"""
This file is adapted from: https://github.com/Algolzw/image-restoration-sde.
Original license: MIT (Copyright © 2023 Ziwei Luo)
Modifications: Incorporation of the additional input mask and Recto-Verso processing
"""
import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils
import cv2

import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default=r"options/test/ir-sde.yml", help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.unlink("./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], 
                 T=opt["sde"]["T"],
                 schedule=opt["sde"]["schedule"], 
                 eps=opt["sde"]["eps"], 
                 device=device)
sde.set_model(model.model)

lpips_fn = lpips.LPIPS(net='alex').to(device)

scale = opt['degradation']['scale']

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["lpips"] = []
    test_times = []

    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ, GT, mask = test_data["LQ"], test_data["GT"], test_data["mask"].to(device)
        noisy_state = sde.noise_state(LQ)

        model.feed_data(torch.cat([noisy_state.to(device), mask], dim=1), LQ, GT)
        tic = time.time()
        model.test(sde)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        SR_imgL, SR_imgR = SR_img[0:3, :, :], SR_img[3:6, :, :]
        outputL = util.tensor2img(SR_imgL)  # uint8
        outputR = util.tensor2img(SR_imgR)  # uint8

        LQ_img = visuals["Input"]
        LQ_L, LQ_R = LQ_img.chunk(2, dim=0)
        LQ_imgL = util.tensor2img(LQ_L.squeeze())
        LQ_imgR = util.tensor2img(LQ_R.squeeze())

        GT_tensor = visuals["GT"]
        GT_tensorL, GT_tensorR = GT_tensor.chunk(2, dim=0)
        GT_imgL = util.tensor2img(GT_tensorL.squeeze())  # uint8
        GT_imgR = util.tensor2img(GT_tensorR.squeeze())  # uint8

        suffix = opt["suffix"]
        if suffix:
            save_imgL_path = os.path.join(dataset_dir, img_name + suffix + ".png")
            save_imgR_path = os.path.join(dataset_dir, img_name.replace('R', 'V') + suffix + ".png")
        else:
            save_imgL_path = os.path.join(dataset_dir, img_name + ".png")
            save_imgR_path = os.path.join(dataset_dir, img_name.replace('R', 'V') + ".png")
        util.save_img(outputL, save_imgL_path)
        util.save_img(outputR, save_imgR_path)

        # remove it if you only want to save output images
        LQ_img_path_L = os.path.join(dataset_dir, img_name + "_LQ.png")
        GT_img_path_L = os.path.join(dataset_dir, img_name + "_HQ.png")
        util.save_img(LQ_imgL, LQ_img_path_L)
        util.save_img(GT_imgL, GT_img_path_L)

        LQ_img_path_R = os.path.join(dataset_dir, img_name.replace('R', 'V') + "_LQ.png")
        GT_img_path_R = os.path.join(dataset_dir, img_name.replace('R', 'V') + "_HQ.png")
        util.save_img(LQ_imgR, LQ_img_path_R)
        util.save_img(GT_imgR, GT_img_path_R)

        if need_GT:
            psnr = util.calculate_psnr(outputL, GT_imgL)
            ssim = util.calculate_ssim(outputL, GT_imgL)
            lp_score = lpips_fn(
                GT_tensorL.to(device) * 2 - 1, SR_imgL.to(device) * 2 - 1).squeeze().item()

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["lpips"].append(lp_score)

            logger.info(
                "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}, LPIPS: {:.6f}.".format(
                    img_name, psnr, ssim, lp_score
                )
            )

            psnr = util.calculate_psnr(outputR, GT_imgR)
            ssim = util.calculate_ssim(outputR, GT_imgR)
            lp_score = lpips_fn(
                GT_tensorR.to(device) * 2 - 1, SR_imgR.to(device) * 2 - 1).squeeze().item()

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["lpips"].append(lp_score)

            logger.info(
                "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}, LPIPS: {:.6f}.".format(
                    img_name.replace('_L', '_R'), psnr, ssim, lp_score
                )
            )

        else:
            logger.info(img_name)

        # break

    ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim
        )
    )

    logger.info(
            "----average LPIPS\t: {:.6f}\n".format(ave_lpips)
        )

    print(f"average test time: {np.mean(test_times):.4f}")

