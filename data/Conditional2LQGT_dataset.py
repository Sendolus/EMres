"""
This file is adapted from: https://github.com/Algolzw/image-restoration-sde.
Original license: MIT (Copyright © 2023 Ziwei Luo)
Modifications: extended to load and return two additional conditions (input masks).
"""
import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class Conditional2LQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths, self.mask_note_paths, self.mask_staff_paths = None, None, None, None
        self.LR_env, self.GT_env, self.mask_note_env, self.mask_staff_env = None, None, None, None  # environment for lmdb
        self.LR_size, self.GT_size = opt["LR_size"], opt["GT_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.LR_paths, self.LR_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
            self.mask_note_paths, self.mask_note_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_mask_note"]
            )
            self.mask_staff_paths, self.mask_staff_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_mask_staff"]
            )
        elif opt["data_type"] == "img":
            self.LR_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
            self.mask_note_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_mask_note"]
            )
            self.mask_staff_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_mask_staff"]
            )
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.GT_paths)
            )
        if self.LR_paths and self.mask_note_paths:
            assert len(self.LR_paths) == len(
                self.mask_note_paths
            ), "mask and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.mask_note_paths)
            )
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.LR_env = lmdb.open(
            self.opt["dataroot_LQ"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.mask_note_env = lmdb.open(
            self.opt["dataroot_mask_note"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.mask_staff_env = lmdb.open(
            self.opt["dataroot_mask_staff"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if (self.GT_env is None) or (self.LR_env is None) or (self.mask_note_env is None) or (self.mask_staff_env is None):
                self._init_lmdb()

        GT_path, LR_path, mask_note_path, mask_staff_path = None, None, None, None
        scale = self.opt["scale"] if self.opt["scale"] else 1
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        # get GT image
        GT_path = self.GT_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        img_GT = util.read_img(
            self.GT_env, GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            img_GT = util.modcrop(img_GT, scale)

        # get LR image and mask
        if self.LR_paths:  # LR exist
            LR_path = self.LR_paths[index]
            mask_note_path = self.mask_note_paths[index]
            mask_staff_path = self.mask_staff_paths[index]
            if self.opt["data_type"] == "lmdb":
                resolution = [int(s) for s in self.LR_sizes[index].split("_")]
            else:
                resolution = None
            img_LR = util.read_img(self.LR_env, LR_path, resolution)
            img_mask_note = util.read_img(self.mask_note_env, mask_note_path, resolution)
            img_mask_staff = util.read_img(self.mask_staff_env, mask_staff_path, resolution)
            # Force masks to grayscale
            if img_mask_note.ndim == 3 and img_mask_note.shape[2] == 3:
                img_mask_note = img_mask_note[:, :, 0]
            if img_mask_staff.ndim == 3 and img_mask_staff.shape[2] == 3:
                img_mask_staff = img_mask_staff[:, :, 0]
        else:  # down-sampling on-the-fly (NOT IMPLEMENTED MASK HERE)
            # randomly scale during training
            if self.opt["phase"] == "train":
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(
                    np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR
                )
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LR = util.imresize(img_GT, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt["phase"] == "train":
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
            img_mask_note = img_mask_note[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size]
            img_mask_staff = img_mask_staff[rnd_h: rnd_h + LR_size, rnd_w: rnd_w + LR_size]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[
                rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
            ]

            # augmentation - flip, rotate
            img_LR, img_GT, img_mask_note, img_mask_staff = util.augment(
                [img_LR, img_GT, img_mask_note, img_mask_staff],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
                self.opt["use_swap"],
            )
        elif LR_size is not None:
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"

            if LR_size < H and LR_size < W:
                # center crop
                rnd_h = H // 2 - LR_size//2
                rnd_w = W // 2 - LR_size//2
                img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[
                    rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
                ]

        # change color space if necessary
        if self.opt["color"]:
            H, W, C = img_LR.shape
            img_LR = util.channel_convert(C, self.opt["color"], [img_LR])[
                0
            ]
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[
                0
            ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()

        # Note masks and staff masks are grayscale and need to be handled differently
        img_mask_note = np.expand_dims(img_mask_note, axis=0)
        img_mask_note = torch.from_numpy(np.ascontiguousarray(img_mask_note)).float()
        img_mask_staff = np.expand_dims(img_mask_staff, axis=0)
        img_mask_staff = torch.from_numpy(np.ascontiguousarray(img_mask_staff)).float()

        if LR_path is None:
            LR_path = GT_path

        return {"LQ": img_LR, "GT": img_GT, "mask_note": img_mask_note, "mask_staff": img_mask_staff,
                "LQ_path": LR_path, "GT_path": GT_path, "mask_note_path": mask_note_path, "mask_staff_path": mask_staff_path}

    def __len__(self):
        return len(self.GT_paths)
