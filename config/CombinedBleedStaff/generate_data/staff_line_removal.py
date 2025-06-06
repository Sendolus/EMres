"""
This script iterates over all page images in a folder and creates versions without staff lines
"""
import pickle
import cv2
import sys
import numpy as np
import glob
import os
from HelperFunctions import staff_line_removal

# Specify the path to the OMR data
directory = r"dataset/EVERYTHING/res_dict"

# Iterate over files in directory
for name in os.listdir(directory):
    print(directory+"/"+name)

    # Load the metadata
    parsed_pickle_name = directory+"/"+name
    with open(parsed_pickle_name, 'rb') as f:
        parsed_folium_dict = pickle.load(f)

    # show the image
    if len(parsed_folium_dict) > 0:
        image_path = parsed_folium_dict[0]['image path']
        try:
            image_no_staff = staff_line_removal(parsed_folium_dict, image_path)
            image_BRG = cv2.cvtColor(image_no_staff, cv2.COLOR_RGB2BGR)
            # Specify output path here
            cv2.imwrite("dataset/EVERYTHING/images_nostaff_part/"+image_name+".jpeg", image_BRG)
        except Exception as e:
            print(f"Error processing {name}: {e}")

