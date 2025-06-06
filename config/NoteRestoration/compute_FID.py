"""
This script can be used to compute FID scores based on a folder of restored images
"""
import os
import shutil
import torch
from pytorch_fid import fid_score
from PIL import Image


def prepare_fid_folders(dataset_dir, gt_folder, sr_folder):
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(sr_folder, exist_ok=True)

    for filename in os.listdir(dataset_dir):
        full_path = os.path.join(dataset_dir, filename)
        if filename.endswith("_HQ.png"):
            shutil.copy(full_path, os.path.join(gt_folder, filename))
        elif filename.endswith(".png") and not filename.endswith("_HQ.png") and not filename.endswith("_LQ.png"):
            shutil.copy(full_path, os.path.join(sr_folder, filename))
    print("Files copied.")


def compute_fid(gt_folder, sr_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_value = fid_score.calculate_fid_given_paths(
        [gt_folder, sr_folder],
        batch_size=50,
        device=device,
        dims=2048,
    )
    print(f"FID: {fid_value:.4f}")


def resize_images(folder, size=(33, 100)):
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            img = Image.open(path).convert("RGB")
            img = img.resize(size, Image.BILINEAR)
            img.save(path)


if __name__ == "__main__":
    dataset_dir = "result/Test/Test_Dataset"
    gt_folder = "FID/GT"
    restored_folder = "FID/Restored"

    prepare_fid_folders(dataset_dir, gt_folder, sr_folder)
    resize_images(gt_folder)
    resize_images(sr_folder)
    compute_fid(sr_folder, gt_folder)
