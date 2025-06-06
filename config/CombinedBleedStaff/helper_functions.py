"""
This file contains helper functions to the Combined model that handles both ink bleedthrough and faded staff lines.
"""
import cv2
import sys
import numpy as np
import glob
import math
import statistics
import os
import torch

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import collections as mc
import matplotlib.image as mpimg
import matplotlib
from get_label_dict import get_dict_of_label_ids
from skimage.exposure import match_histograms

sys.path.insert(0, "../../")
import utils as util

label_dict = get_dict_of_label_ids()
inv_label_dict = {v: k for k, v in label_dict.items()}


def inference_page_rv(parsed_folium, image, verso, rmask, vmask, rsmask, vsmask, ref_img, model, sde, stop_early):
    """
    Function that processes full pages images.

    input parameters:
    - parsed_folium : output file from the adjacent OMR model
    - image         : image of the recto page (possibly rescaled)
    - verso         : image of the verso page, alligned with the recto image (possibly rescaled)
    - rmask         : mask indicating the estimated positions of the recto staffs
    - vmask         : mask indicating the estimated positions of the verso staffs
    - rsmask        : mask indicating the estimated positions of the recto staff lines
    - vsmask        : mask indicating the estimated positions of the verso staff lines
    - ref_img       : image of the recto side in original resolution
    - model         : Neural network model
    - sde           : SDE-object governing the diffusion process
    - stop_early    : integer that the last 'stop_early' staffs don't need to be processed

    outputs:
    - resultR       : restored recto page
    - resultV       : restored verso page
    """

    # Normalize all inputs
    height, width, _ = ref_img.shape
    notes_min_x, notes_max_x, notes_min_y, notes_max_y = determine_note_area(parsed_folium)
    notes_width = notes_max_x - notes_min_x
    scaling_factor_w = 500 / (notes_width )
    scaling_factor_h = 250 / (3*parsed_folium[0]['staff_height'])
    new_width = int(width * scaling_factor_w)
    new_height = int(height * scaling_factor_h)
    notes_min_x = int(notes_min_x * scaling_factor_w)
    notes_max_x = int(notes_max_x * scaling_factor_w)
    img_rescaled = cv2.resize(image, (new_width, new_height))
    verso_rescaled = cv2.resize(verso, (new_width, new_height))
    rmask_rescaled = cv2.resize(rmask, (new_width, new_height))
    vmask_rescaled = cv2.resize(vmask, (new_width, new_height))
    rsmask_rescaled = cv2.resize(rsmask, (new_width, new_height))
    vsmask_rescaled = cv2.resize(vsmask, (new_width, new_height))

    # Loop over the staffs to perform patchwise processing
    nb_staffs = len(parsed_folium)
    j = 0
    for staff_dict in parsed_folium:
        if -1 < j < nb_staffs - stop_early:

            # Extract staff location
            stave_min_x = 1e6
            stave_max_x = 0
            stave_min_y = 1e6
            stave_max_y = 0
            labels = [inv_label_dict[label_id] for label_id in staff_dict['label_ids']]
            staff_height = int(staff_dict['staff_height'] * scaling_factor_h)
            for i in range(len(labels)):
                obj = staff_dict['boxes'][i]
                x, y, w, h = obj[1], obj[0], obj[3] - obj[1], obj[2] - obj[0]
                xmin = int(x * scaling_factor_w)
                xmax = int((x + w) * scaling_factor_w)
                ymin = int(y * scaling_factor_h)
                ymax = int((y + h) * scaling_factor_h)
                if xmin < stave_min_x:
                    stave_min_x = xmin
                if xmax > stave_max_x:
                    stave_max_x = xmax
                if ymin < stave_min_y:
                    stave_min_y = ymin
                if ymax > stave_max_y:
                    stave_max_y = ymax

            # Cut out patches
            patch = img_rescaled[max(stave_min_y, 0):min(stave_max_y + 2 * staff_height, new_height),max(notes_min_x, 0):min(notes_max_x, new_width)]
            patch_verso = verso_rescaled[max(stave_min_y, 0):min(stave_max_y + 2 * staff_height, new_height),max(notes_min_x, 0):min(notes_max_x, new_width)]
            patch_rmask = rmask_rescaled[max(stave_min_y, 0):min(stave_max_y + 2 * staff_height, new_height),max(notes_min_x, 0):min(notes_max_x, new_width)]
            patch_vmask = vmask_rescaled[max(stave_min_y, 0):min(stave_max_y + 2 * staff_height, new_height),max(notes_min_x, 0):min(notes_max_x, new_width)]
            patch_rsmask = rsmask_rescaled[max(stave_min_y, 0):min(stave_max_y + 2 * staff_height, new_height),max(notes_min_x, 0):min(notes_max_x, new_width)]
            patch_vsmask = vsmask_rescaled[max(stave_min_y, 0):min(stave_max_y + 2 * staff_height, new_height),max(notes_min_x, 0):min(notes_max_x, new_width)]

            # Blur areas that are only on one side
            mask1_interest = patch_rmask > 127
            mask2_interest = patch_vmask > 127
            blur_mask_r = np.logical_and(mask2_interest, np.logical_not(mask1_interest)).astype(np.uint8) * 255
            blur_mask_v = np.logical_and(np.logical_not(mask2_interest), mask1_interest).astype(np.uint8) * 255
            sigma_x = 1
            sigma_y = 0.2
            patch_blurred = cv2.GaussianBlur(patch, ksize=(5, 5), sigmaX=sigma_x, sigmaY=sigma_y)
            patch_verso_blurred = cv2.GaussianBlur(patch_verso, ksize=(5, 5), sigmaX=sigma_x, sigmaY=sigma_y)
            maskr_norm = (blur_mask_r > 127).astype(np.float32)
            maskv_norm = (blur_mask_v > 127).astype(np.float32)
            patch = patch * (1 - maskr_norm) + patch_blurred * maskr_norm
            patch = np.array(patch, dtype=np.uint8)
            patch_verso = patch_verso * (1 - maskv_norm) + patch_verso_blurred * maskv_norm
            patch_verso = np.array(patch_verso, dtype=np.uint8)

            # Feed the patches to the diffusion model for restoration
            restored_patch_recto, restored_patch_verso = diffusion_inference_rv(patch, patch_verso, patch_rsmask, patch_vsmask, model, sde)

            # Trim off bottom to avoid unknown glitching effect
            restored_patch_recto = restored_patch_recto[0: -8, :]
            restored_patch_verso = restored_patch_verso[0: -8, :]

            # POSTPROCESSING: stitch patches back into the full page
            # Define blending coefficients in feathered mask
            mask = np.zeros_like(restored_patch_recto)
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[1:-1, 1:-1] = 255
            dist = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
            feathered_mask = dist / 30
            feathered_mask[30:-30, 30:-30] = 1
            feathered_mask = feathered_mask ** 1.5
            feathered_mask = feathered_mask[..., None]

            # Perform the actual blending
            center = (int((max(notes_min_x, 0) + min(notes_max_x, new_width)) / 2), int((max(stave_min_y, 0) + min(stave_max_y + 2 * staff_height, new_height)) / 2) - 4)
            x, y = center
            x -= restored_patch_recto.shape[1] // 2
            y -= restored_patch_recto.shape[0] // 2
            h, w = restored_patch_recto.shape[:2]
            target_region = img_rescaled[y:y + h, x:x + w].astype(np.float32)
            patch_float = restored_patch_recto.astype(np.float32)
            blended = patch_float * feathered_mask + target_region * (1 - feathered_mask)

            target_region = verso_rescaled[y:y + h, x:x + w].astype(np.float32)
            patch_float_verso = restored_patch_verso.astype(np.float32)
            blended_verso = patch_float_verso * feathered_mask + target_region * (1 - feathered_mask)

            # Paste result back into image
            img_rescaled[y:y + h, x:x + w] = blended.astype(np.uint8)
            verso_rescaled[y:y + h, x:x + w] = blended_verso.astype(np.uint8)

            j += 1

    img_rescaled = cv2.cvtColor(img_rescaled, cv2.COLOR_BGR2RGB)
    verso_rescaled = cv2.cvtColor(verso_rescaled, cv2.COLOR_BGR2RGB)

    resultR = cv2.resize(img_rescaled, (width, height), interpolation=cv2.INTER_AREA)
    resultV = cv2.resize(verso_rescaled, (width, height), interpolation=cv2.INTER_AREA)
    return resultR, resultV


def diffusion_inference_rv(image, verso, rmask, vmask, model, sde):
    """
    Function that feeds recto and verso image patches to the diffusion model.

    input parameters:
    - image         : image of the recto patch
    - verso         : image of the verso patch
    - rmask         : mask indicating the estimated positions of the recto staff lines
    - vmask         : mask indicating the estimated positions of the verso staff lines
    - model         : Neural network model
    - sde           : SDE-object governing the diffusion process

    outputs:
    - outputL       : restored recto page
    - outputR       : restored verso page
    """
    # Convert the images (numpy) to tensors
    image = image[:, :, [2, 1, 0]] / 255.0
    verso = verso[:, :, [2, 1, 0]] / 255.0
    rmask = rmask[..., 0] / 255.0
    vmask = vmask[..., 0] / 255.0
    image_torch = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float().unsqueeze(0)
    verso_torch = torch.from_numpy(np.ascontiguousarray(np.transpose(verso, (2, 0, 1)))).float().unsqueeze(0)
    rmask = np.squeeze(rmask)
    rmask = np.expand_dims(rmask, axis=0)
    rmask = torch.from_numpy(np.ascontiguousarray(rmask)).float().unsqueeze(0)
    vmask = np.squeeze(vmask)
    vmask = np.expand_dims(vmask, axis=0)
    vmask = torch.from_numpy(np.ascontiguousarray(vmask)).float().unsqueeze(0)
    GT = torch.cat([image_torch, verso_torch], dim=1)
    LQ = torch.cat([image_torch, verso_torch], dim=1)

    # Generate noisy starting state and feed this to the diffusion model
    noisy_state = sde.noise_state(LQ)
    model.feed_data(torch.cat([noisy_state, rmask, vmask], dim=1), LQ, GT)
    model.test(sde, save_states=True)

    # Return the results
    visuals = model.get_current_visuals()
    SR_img = visuals["Output"]
    SR_imgR, SR_imgV = SR_img[:3, :, :], SR_img[3:6, :, :]
    outputR = util.tensor2img(SR_imgR.squeeze())  # uint8
    outputV = util.tensor2img(SR_imgV.squeeze())
    return outputR, outputV


def determine_note_area(parsed_folium):
    """
    Function that determines the area of the page that contains foreground notes.

    input parameters:
    - parsed_folium : output file from the adjacent OMR model

    outputs:
    - stave_min_x   : determines left border of note area
    - stave_max_x   : determines right border of note area
    - stave_min_y   : determines top border of note area
    - stave_max_y   : determines bottom border of note area
    """
    stave_min_x = 1e6
    stave_max_x = 0
    stave_min_y = 1e6
    stave_max_y = 0

    for staff_dict in parsed_folium:
        labels = [inv_label_dict[label_id] for label_id in staff_dict['label_ids']]
        for i in range(len(labels)):
            obj = staff_dict['boxes'][i]
            x, y, w, h = obj[1], obj[0], obj[3] - obj[1], obj[2] - obj[0]
            xmin = x
            xmax = x + w
            ymin = y
            ymax = y + h
            if xmin < stave_min_x:
                stave_min_x = xmin
            if xmax > stave_max_x:
                stave_max_x = xmax
            if ymin < stave_min_y:
                stave_min_y = ymin
            if ymax > stave_max_y:
                stave_max_y = ymax

    return stave_min_x, stave_max_x, stave_min_y, stave_max_y


def align_pages_by_hand(recto_orig, verso_orig):
    """
    Tool to manually align recto and verso page sides.

    input parameters:
    - recto_orig        : image of recto page
    - verso_orig        : image of verso page

    outputs:
    - aligned_full_res  : aligned verso page
    """
    H_orig, W_orig = verso_orig.shape[0], verso_orig.shape[1]
    recto = cv2.resize(recto_orig, (720, 1080))
    verso = cv2.resize(verso_orig, (720, 1080))
    points_recto = []
    points_verso = []
    new_points_verso = []
    click_stage = 0  # 0 = recto, 1 = verso

    def click_points(event, x, y, flags, param):
        nonlocal click_stage
        if event == cv2.EVENT_LBUTTONDOWN:
            if click_stage % 2 == 0:
                points_recto.append([x, y])
                print(f"Recto point {len(points_recto)}: {x}, {y}")
            else:
                points_verso.append([x, y])
                print(f"Verso point {len(points_verso)}: {x}, {y}")
            click_stage += 1

    cv2.namedWindow("Image Selection")
    cv2.setMouseCallback("Image Selection", click_points)
    print("Click a point on RECTO, then its match on VERSO, then next RECTO, etc... Press ENTER when done.")

    while True:
        combined = np.hstack((recto.copy(), verso.copy()))
        offset = recto.shape[1]

        for i, pt in enumerate(points_recto):
            cv2.circle(combined, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(combined, f"R{i + 1}", tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for i, pt in enumerate(points_verso):
            verso_pt = (pt[0], pt[1])
            cv2.circle(combined, tuple(verso_pt), 5, (0, 255, 0), -1)
            cv2.putText(combined, f"V{i + 1}", tuple(verso_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Recto verso alignment", combined)
        key = cv2.waitKey(1)
        if key == 13:  # ENTER key
            break

    cv2.destroyAllWindows()

    # Adjust for offset
    for point in points_verso:
        test = [point[0] - offset, point[1]]
        new_points_verso.append(test)
    points_verso = new_points_verso

    if 4 <= len(points_verso) == len(points_recto):
        # Homography and warp resized image to display
        pts1 = np.array(points_recto, dtype=np.float32)
        pts2 = np.array(points_verso, dtype=np.float32)
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
        aligned = cv2.warpPerspective(verso, H, (recto.shape[1], recto.shape[0]))
        blended = cv2.addWeighted(aligned, 0.5, recto, 0.5, 0)
        cv2.imshow("Aligned & Blended", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Also warp the original image (full resolution)
        scale_x = W_orig / 720
        scale_y = H_orig / 1080
        S = np.array([[scale_x, 0, 0],
                      [0, scale_y, 0],
                      [0, 0, 1]])
        H_original = S @ H @ np.linalg.inv(S)
        aligned_full_res = cv2.warpPerspective(verso_orig, H_original, (recto_orig.shape[1], recto_orig.shape[0]))
        return aligned_full_res

    else:
        print("Not enough matching points (need at least 4), or point count mismatch.")
