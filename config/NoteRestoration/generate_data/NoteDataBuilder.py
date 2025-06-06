"""
This script is used to generate synthetic data for the training set
"""
import sys
import cv2
import os
import random
import pickle
from HelperFunctions import *

# Specify the paths to the image folder (full pages), the OMR output (pickle folder), and the output folder
image_folder = "dataset/EVERYTHING/images"
pickle_folder = "dataset/EVERYTHING/res_dict"
output_folder = "dataset/Notes_testV3_big"

pickle_files = sorted(os.listdir(pickle_folder))  # Sort to ensure consistent order
nb_img = len(pickle_files)

for i in range(5000):
    file = pickle_files[np.random.randint(0, nb_img)]
    pickle_path = os.path.join(pickle_folder, file)
    try:
        with open(pickle_path, 'rb') as f:
            parsed_folium_dict = pickle.load(f)

        img_path = parsed_folium_dict[0]['image path']
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]
        img_path = "\\".join(img_path.split("/")[-1:])
        img_path = image_folder + "\\" + img_path
        LQ, HQ, Context = make_note_training(parsed_folium_dict, img_path, random.uniform(-0.2, 0.2))

        LQ_path = os.path.join(output_folder, "LQ", str(i) + ".jpeg")
        HQ_path = os.path.join(output_folder, "HQ", str(i) + ".jpeg")
        cv2.imwrite(LQ_path, LQ)
        cv2.imwrite(HQ_path, HQ)

        for j in range(len(Context)):
            Context_path = os.path.join(output_folder, "Context", str(i) + "_" + str(j) + ".jpeg")
            cv2.imwrite(Context_path, Context[j])

        print(f"Processed: {i}")

    except Exception as e:
        print(f"Error processing {file}: {e}")


print("Building data completed!")

