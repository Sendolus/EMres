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


class ContextLQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths, self.context_paths = None, None, None
        self.LR_env, self.GT_env, self.context_env = None, None, None  # environment for lmdb
        self.LR_size, self.GT_size = opt["LR_size"], opt["GT_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.LR_paths, self.LR_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
            self.context_paths, self.context_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_context"]
            )
        elif opt["data_type"] == "img":
            self.LR_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
            self.context_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_context"]
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
        if self.LR_paths and self.context_paths:
            assert 3 * len(self.LR_paths) == len(
                self.context_paths
            ), "context and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.context_paths)
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
        self.context_env = lmdb.open(
            self.opt["dataroot_context"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if (self.GT_env is None) or (self.LR_env is None) or (self.context_env is None):
                self._init_lmdb()

        GT_path, LR_path, context_path = None, None, None
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

        # get LR image and context images
        if self.LR_paths:  # LR exist
            LR_path = self.LR_paths[index]

            base_dir = os.path.dirname(os.path.dirname(LR_path))  # => "datasets/Notes_trainV2"
            context_dir = os.path.join(base_dir, "Context")  # => "datasets/Notes_trainV2/Context"
            filename = os.path.basename(LR_path)  # => "3272.jpeg"
            file_stem = os.path.splitext(filename)[0]  # => "3272"

            context1_path = os.path.join(context_dir, f"{file_stem}_0.jpeg")
            context2_path = os.path.join(context_dir, f"{file_stem}_1.jpeg")
            context3_path = os.path.join(context_dir, f"{file_stem}_2.jpeg")
            if self.opt["data_type"] == "lmdb":
                resolution = [int(s) for s in self.LR_sizes[index].split("_")]
            else:
                resolution = None
            img_LR = util.read_img(self.LR_env, LR_path, resolution)
            img_context1 = util.read_img(self.context_env, context1_path, resolution)
            img_context2 = util.read_img(self.context_env, context2_path, resolution)
            img_context3 = util.read_img(self.context_env, context3_path, resolution)
        else:  # down-sampling on-the-fly (NOT IMPLEMENTED context HERE)
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
            img_LR = img_LR[rnd_h: rnd_h + LR_size, rnd_w: rnd_w + LR_size, :]
            img_context1 = img_context1[rnd_h: rnd_h + LR_size, rnd_w: rnd_w + LR_size]
            img_context2 = img_context2[rnd_h: rnd_h + LR_size, rnd_w: rnd_w + LR_size]
            img_context3 = img_context3[rnd_h: rnd_h + LR_size, rnd_w: rnd_w + LR_size]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[
                rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
            ]

            # augmentation - flip, rotate
            img_LR, img_GT, img_context1, img_context2, img_context3 = util.augment(
                [img_LR, img_GT, img_context1, img_context2, img_context3],
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
            ]  # TODO during val no definition
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[
                0
            ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
            img_context1 = img_context1[:, :, [2, 1, 0]]
            img_context2 = img_context2[:, :, [2, 1, 0]]
            img_context3 = img_context3[:, :, [2, 1, 0]]

        img_GT = cv2.resize(img_GT, (33, 100))
        img_LR = cv2.resize(img_LR, (33, 100))
        img_context1 = cv2.resize(img_context1, (33, 100))
        img_context2 = cv2.resize(img_context2, (33, 100))
        img_context3 = cv2.resize(img_context3, (33, 100))

        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()
        img_context1 = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_context1, (2, 0, 1)))
        ).float()
        img_context2 = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_context2, (2, 0, 1)))
        ).float()
        img_context3 = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_context3, (2, 0, 1)))
        ).float()
        if LR_path is None:
            LR_path = GT_path

        return {"LQ": img_LR, "GT": img_GT, "context1": img_context1, "context2": img_context2,
                "context3": img_context3, "LQ_path": LR_path, "GT_path": GT_path, "context1_path": context1_path,
                "context2_path": context2_path, "context3_path": context3_path}

    def __len__(self):
        return len(self.GT_paths)
