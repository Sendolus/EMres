import sys
import cv2
import os
import pickle
from HelperFunctions import *

# Specify paths to input and output folders
pickle_folder = "dataset/alamire_bad_res_dict"      # Contains images
output_folder = "dataset/Alamire_bad/staves"        # Where masks are storded
i = 0

for pickle_file in os.listdir(pickle_folder):
    pickle_path = os.path.join(pickle_folder, pickle_file)
    try:
        i += 1
        with open(pickle_path, 'rb') as f:
            parsed_folium_dict = pickle.load(f)

        image_path = parsed_folium_dict[0]['image path']
        mask = generate_masks(parsed_folium_dict, image_path)
        img_name = image_path.split("\\")[-1].rsplit(".", 1)[0]
        output_path = output_folder + "/" + f"{img_name}.jpeg"
        cv2.imwrite(output_path, mask)

        print(f"¨Processed: {pickle_file}, i = {i}")

    except Exception as e:
        print(f"Error processing {pickle_file}: {e}")

print("Building training data completed!")