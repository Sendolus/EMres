"""
This file is used to generate synthetic training data. It requires the installation of the Augraphy library
"""
import sys
import cv2
import os
import random
from InkBleedThrough import *
from augraphy import *

# Specify paths to image folders
image_folder = "dataset/EVERYTHING/triplets/train_val_test/train/Orig"
image_folder_ns = "dataset/EVERYTHING/triplets/train_val_test/train/NoStaff"   # images with staff lines removed
mask_folder = "dataset/EVERYTHING/triplets/train_val_test/train/MaskStaffline"

# Specify output paths where the data is saved
train_folder = "dataset/FinalComboSequential/train/LQ"
hr_folder = "dataset/FinalComboSequential/train/HQ"
mask_save_folder = "dataset/FinalComboSequential/train/mask"

# Process images pairwise
image_files = sorted(os.listdir(image_folder))  # Sort to ensure consistent order
nb_img = len(image_files)

# ------------- INITIALIZE AUGRAPHY ------------------
ink_phase = [

    InkBleed(
        intensity_range=(0.1, 1),
        kernel_size=random.choice([(5, 5), (3, 3)]),
        severity=(0.1, 2),
        p=0.1
    ),
    ColorShift(
        color_shift_offset_x_range=(3, 5),
        color_shift_offset_y_range=(3, 5),
        color_shift_iterations=(1, 3),
        color_shift_brightness_range=(1, 1),
        color_shift_gaussian_kernel_range=(3, 5),
        p=0.1
    ),
    LinesDegradation(
        line_roi=(0.0, 0.0, 1.0, 1.0),
        line_gradient_range=(32, 255),
        line_gradient_direction=(2, 2),
        line_split_probability=(0.2, 0.3),
        line_replacement_value=(0, 50),
        line_min_length=(10, 10),
        line_long_to_short_ratio=(3, 3),
        line_replacement_probability=(0.5, 0.5),
        line_replacement_thickness=(1, 1),
        p=0.1
    ),
]

paper_phase = []

post_phase = [
    SubtleNoise(
        subtle_range=random.randint(5, 10),
        p=0.2,
    ),

]

pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

# ---------------- START LOOP -------------------

for i in range(2500):
    file = image_files[np.random.randint(0, nb_img)]
    file2 = image_files[np.random.randint(0, nb_img)]

    file_path = os.path.join(image_folder_ns, file)
    file_path2 = os.path.join(image_folder_ns, file2)

    orig_path = os.path.join(image_folder, file)
    orig_path2 = os.path.join(image_folder, file2)
    mask_path = os.path.join(mask_folder, file)
    mask_path2 = os.path.join(mask_folder, file2)

    try:
        # Read images using OpenCV
        image = cv2.imread(file_path)
        image2 = cv2.imread(file_path2)
        orig_image = cv2.imread(orig_path)
        orig_image2 = cv2.imread(orig_path2)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)

        if image is None or image2 is None:
            print(f"Skipping {file} or {file2}: Cannot be read")
            continue  # Skip unreadable files

        # Apply augmentation pipeline with both images
        image_aug = pipeline(image)
        image2_aug = pipeline(image2)

        h = cv2.getGaussianKernel(ksize=5, sigma=random.uniform(1, 3))
        h = np.outer(h, h)  # Make it 2D
        recto, verso = simulate_bleedtrough(image_aug, image2_aug, h, random.uniform(0, 0.60))
        verso = cv2.flip(verso, 1)  # Flip left-to-right

        # Save the modified image
        recto_path = os.path.join(train_folder, str(i)+"_R_.jpg")
        verso_path = os.path.join(train_folder, str(i)+"_V_.jpg")
        recto_hr_path = os.path.join(hr_folder, str(i)+"_R_.jpg")
        verso_hr_path = os.path.join(hr_folder, str(i)+"_V_.jpg")
        recto_mask_path = os.path.join(mask_save_folder, str(i)+"_R_.jpg")
        verso_mask_path = os.path.join(mask_save_folder, str(i)+"_V_.jpg")
        cv2.imwrite(recto_path, recto)
        cv2.imwrite(verso_path, cv2.flip(verso, 1))
        cv2.imwrite(recto_hr_path, orig_image)
        cv2.imwrite(verso_hr_path, orig_image2)
        cv2.imwrite(recto_mask_path, mask)
        cv2.imwrite(verso_mask_path, mask2)
        print(f"Modified: {file}, i = {i}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

print("Building training data completed!")
