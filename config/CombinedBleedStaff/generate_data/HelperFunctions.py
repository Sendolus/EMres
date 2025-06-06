"""
Helper Functions for synthetic data generation
"""
import cv2
import sys
import numpy as np
import glob
import math
import statistics
import os
import itertools
import random
from collections import Counter
from opensimplex import OpenSimplex
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import collections as mc
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib
from get_label_dict import get_dict_of_label_ids
sys.path.append('post_dcnn_inference')
from stave_line_detection import get_stave_line_candidates


def extract_staves_triplets(parsed_folium, image_folder, output_folder):
    """
    Extracts staff triplets form a full page
    """
    img_path = parsed_folium[0]['image path']
    img_name = img_path.split("/")[-1].rsplit(".", 1)[0]
    img_path = "\\".join(img_path.split("/")[-1:])
    img_path = image_folder + img_path
    gray = False
    scale_factor_width, scale_factor_height = 1, 1
    image = rescale_image(img_path, gray, scale_factor_width, scale_factor_height)

    # Get image dimensions and note area
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    notes_min_x, notes_max_x, notes_min_y, notes_max_y = determine_note_area(parsed_folium)

    staff_nb = 0
    for staff_dict in parsed_folium:
        staff_nb += 1
        stave_min_y = 1e6
        stave_max_y = 0
        labels = [inv_label_dict[label_id] for label_id in staff_dict['label_ids']]
        staff_height = staff_dict['staff_height']
        for i in range(len(labels)):
            obj = staff_dict['boxes'][i]
            x, y, w, h = obj[1], obj[0], obj[3] - obj[1], obj[2] - obj[0]
            ymin = y
            ymax = y + h
            if ymin < stave_min_y:
                stave_min_y = ymin
            if ymax > stave_max_y:
                stave_max_y = ymax

        cropped = image[max(stave_min_y,0):min(stave_max_y+2*staff_height,height), max(notes_min_x,0):min(notes_max_x, width)]
        output_path = os.path.join(output_folder, f"{img_name}_crop_{staff_nb}.jpg")
        cropped_BRG = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cropped_BRG)


def rescale_image(img_path, gray, scale_factor_width, scale_factor_height):
    """
    Rescales image
    """
    if gray:
        img = cv2.imread(img_path, 0)
    else:
        img = mpimg.imread(img_path)
    height, width = img.shape[0], img.shape[1]

    if scale_factor_width == None and scale_factor_height == None:
        scale_factor_width = 1
        scale_factor_height = 1
    img = cv2.resize(img, (int((width * scale_factor_width)), int(height * scale_factor_height)),
                     interpolation=cv2.INTER_NEAREST)
    return img


def staff_line_removal(parsed_folium, img_path):
    """
    Remove staff lines from image
    """
    x_extension = 8
    offset = 0
    gray = False
    scale_factor_width, scale_factor_height = 1, 1
    image = rescale_image(img_path, gray, scale_factor_width, scale_factor_height)
    line_width = image.shape[0] // 500
    tmp = OpenSimplex(seed=np.random.randint(0, 10000))

    # Extract lines from OMR
    lines = []
    for staff_dict in parsed_folium:
        labels = [inv_label_dict[label_id] for label_id in staff_dict['label_ids']]
        for i in range(len(labels)):
            obj = staff_dict['boxes'][i]
            x, y, w, h = obj[1], obj[0], obj[3] - obj[1], obj[2] - obj[0]
            xmin = x
            xmax = x + w
            if 'staff_lines_around_music_symbol' in staff_dict.keys():
                staff_lines_around_music_symbol = staff_dict['staff_lines_around_music_symbol']
                for j in [3, 5, 7, 9, 11]:
                    lines.append([(xmin - x_extension, staff_lines_around_music_symbol[i][j] + offset), \
                                  (xmax + x_extension, staff_lines_around_music_symbol[i][j] + offset)])

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for line in lines:
        start, end = line
        cv2.line(mask, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color=255, thickness=line_width)

    # Extract staff lines in areas with no notes through a horizontal filter
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -5)
    horizontal_size = gray.shape[1] // 20
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(bw, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, line_width))
    horizontal = cv2.dilate(horizontal, kernel, iterations=1)
    mask = mask + horizontal

    # Remove part of the mask to leave some staff lines in place (OPTIONAL)
    H, W = mask.shape
    for i in range(H):
        for j in range(W):
            scale_x = 10 / H
            scale_y = 2 / W
            value = tmp.noise2(i * scale_x, j * scale_y)
            if value > 0.2:
                mask[i, j] = 0

    # Use inpainting to remove the found staffs
    image_no_staff = cv2.inpaint(image, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    return image_no_staff


def generate_masks(parsed_folium, img_path):
    """
    generates a mask with estimated staff line positions
    """
    gray = False
    scale_factor_width, scale_factor_height = 1, 1
    image = rescale_image(img_path, gray, scale_factor_width, scale_factor_height)
    mask_staff_lines = np.zeros(image.shape[:2], dtype=np.uint8)
    notes_min_x, notes_max_x, notes_min_y, notes_max_y = determine_note_area(parsed_folium)
    line_width = image.shape[0] // 500

    # Extract lines from OMR
    for index, staff_dict in enumerate(parsed_folium):
        boxes = staff_dict['boxes']
        labels = [inv_label_dict[label_id] for label_id in staff_dict['label_ids']]

        for i in range(0, len(labels)):
            obj = boxes[i]
            x, y, w, h = obj[1], obj[0], obj[3] - obj[1], obj[2] - obj[0]
            xmin = x
            xmax = x + w
            if 'staff_lines_around_music_symbol' in staff_dict.keys():
                staff_lines_around_music_symbol = staff_dict['staff_lines_around_music_symbol']
                for j in [3, 5, 7, 9, 11]:
                    cv2.line(mask_staff_lines, (xmin - 8, int(staff_lines_around_music_symbol[i][j])),
                             (xmax + 8, int(staff_lines_around_music_symbol[i][j])), 255, line_width)

    # Extract staff lines in areas with no notes through a horizontal filter
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -6)
    horizontal_size = (notes_max_x - notes_min_x) // 15
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(bw, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, line_width))
    horizontal = cv2.dilate(horizontal, kernel, iterations=1)

    # FILTER HORIZONTAL LINES BASED ON X-COORDINATES
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_horizontal = np.zeros_like(horizontal)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if notes_min_x - gray.shape[1] // 10 <= x and x+w <= notes_max_x + gray.shape[1] // 10:
            cv2.drawContours(filtered_horizontal, [contour], -1, 255, thickness=cv2.FILLED)
    filtered_horizontal[:, notes_max_x:] = 0
    filtered_horizontal[:, :notes_min_x] = 0

    return mask_staff_lines + filtered_horizontal


def determine_note_area(parsed_folium):
    """
    Function that determines the area of the page that contains foreground notes.
    """
    # Initialize the parameters
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