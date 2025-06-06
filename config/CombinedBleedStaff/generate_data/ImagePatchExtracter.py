"""
Extracts image patches (staff triplets) from full-page images or masks
"""
import sys
import cv2
import os
import pickle
from HelperFunctions import *

# path to OMR outputs
pickle_folder = r"dataset/Alamire_bad_res_dict"
i = 0

for pickle_file in os.listdir(pickle_folder):
    pickle_path = os.path.join(pickle_folder, pickle_file)
    try:
        i += 1
        with open(pickle_path, 'rb') as f:
            parsed_folium_dict = pickle.load(f)

        # show the image
        image_folder = ""
        extract_staves_triplets(parsed_folium_dict, image_folder, "dataset/Alamire/images/triplets")

        print(f"Modified: {pickle_file}, i = {i}")

    except Exception as e:
        print(f"Error processing {pickle_file}: {e}")

print("Building training data completed!")

