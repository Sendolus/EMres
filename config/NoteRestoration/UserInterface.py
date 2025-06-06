"""
Running this file opens up the user interface to restore damaged notes
"""
import pickle
import argparse
import cv2
import sys
import numpy as np
import glob
import os

import torch.nn.functional as F
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import collections as mc
import matplotlib.image as mpimg
from get_label_dict import get_dict_of_label_ids
from helper_functions import *
from matplotlib.widgets import Button, TextBox
import matplotlib
import options as option
from models import create_model

# Specify the path to the image to which note restoration should be applied
image_path = r""
# Specify the path to the OMR output
parsed_pickle_name = r'datasets/res_dict_bleedstaff/89_bleed_staff89_bleed_staff.pickle'
with open(parsed_pickle_name, 'rb') as f:
    parsed_folium_dict = pickle.load(f)

color_dictionary = { 'minim': 'black', 'semibreve': 'crimson', 'obliqa': 'bisque', 'semiminim': 'violet',
                   'dot': 'fuchsia', 'dotted note': 'brown', 'rest': 'lime',     'breve': 'navy',
                   'barline': 'yellow', 'repeat': 'slategray',  'fermata': 'yellow',
                   'ligature':  'teal',    'lnote1': 'chocolate', 'lnote2': 'lavenderblush',
                   'l1colored breve': 'cyan','imin': 'mediumvioletred',   'flat': 'blue',
                    'fusa': 'darkorange', 'longa': 'slateblue','colored breve' : 'cyan',
                   'colored semibreve': 'lavenderblush', 'colored longa' : 'lightcyan',  'c clef': 'olivedrab',
                   'f clef': 'mintcream',  'g clef': 'lightcoral',  'imaj': 'darkorange',
                   'pmaj': 'darkkhaki', 'imincut': 'darkgreen', 'imincut': 'chocolate',
                   'pmin': 'saddlebrown', '3': 'mediumpurple',  '2': 'aliceblue', 'sharp': 'gold',
                   'fusa rest': 'cyan', 'semifusa':'aquamarine', '1': 'brown','custos' : 'teal',
                   'minim rest': 'teal', 'breve rest': 'pink', 'semibreve rest': 'yellowgreen',
                   'longa rest': 'slateblue', 'colored lnote1' :'green', 'colored lnote2': 'cyan',
                   'colored obliqa': 'yellow', 'white_black obliqa': 'purple', 'black_white obliqa':
                   'violet', 'pmincut': 'gray','l2colored breve': 'red','l1semibreve': 'crimson' ,
                  'l2semibreve': 'crimson', 'l1breve': 'navy', 'l2breve': 'navy',
                   'congruence': 'brown', 'o1semibreve': 'red', 'o2semibreve':'crimson',
                   'l1colored semibreve' : 'lavenderblush', 'o1colored semibreve': 'lavenderblush', 'o2breve' :'navy',
                   'o2colored semibreve' : 'lavenderblush','l1longa' : 'slateblue', 'o1breve': 'navy', 'o1colored breve':'cyan',
                  'l2colored semibreve' : 'lavenderblush', 'l2longa': 'slateblue', 'ornate element': 'purple',
                  'o2colored breve': 'cyan', 'l1': 'brown','l2': 'white', 'o1': 'red', 'o2': 'blue', 'colored l1': 'pink',
                  'colored l2': 'cyan', 'colored o1' : 'teal', 'colored o2': 'white',
                   'o1longa': 'slateblue' ,'o2longa': 'slateblue','l2colored longa': 'lightcyan','l1colored longa': 'lightcyan' }

# Load diffusion model:
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default=r"options/test/ir-sde.yml", help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)
model = create_model(opt)
device = model.device
sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)

# Let the user click damaged notes
test_image = cv2.imread(image_path)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
damaged_notes = click_damaged_notes(parsed_folium_dict, image_path)

# Process each note that was clicked sequentially
for (index, label) in damaged_notes:
    # Find clean context note examples
    clean_notes = find_good_notes(parsed_folium_dict, image_path, label)
    note_img = note_id2image(parsed_folium_dict, image_path, index)
    note_backup = note_img.copy()
    total_images = 4
    staff, pos = index.split(":")
    box = parsed_folium_dict[int(staff)]['boxes'][int(pos)]

    # Figure size and layout
    fig_width = 8
    fig_height = 4
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = []
    buttons = []

    # Callback for flip button
    def make_flip_callback(ax_img):
        def flip(event):
            current_img = ax_img.images[0].get_array()
            flipped_img = np.flipud(current_img)
            ax_img.images[0].set_data(flipped_img)
            fig.canvas.draw()
        return flip

    # Callback for change button
    def make_change_callback(ax_img):
        def change(event):
            def on_note_selected(note_nb, label):
                print(f"Selected: {note_nb}, {label}")
                new_note_img = note_id2image(parsed_folium_dict, image_path, note_nb)
                ax_img.images[0].set_data(new_note_img)
                fig.canvas.draw()

            click_one_note(parsed_folium_dict, image_path, on_note_selected)
        return change

    # Callback for 'Paint' button
    def make_paint_callback(ax_img):
        def paint(event):
            current_img = ax_img.images[0].get_array()
            painted_img = paint_on_img(current_img)
            ax_img.images[0].set_data(painted_img)
            fig.canvas.draw()
        return paint

    # Callback for 'Edit area' button
    def make_edit_area_callback(ax_img):
        def edit_area(event):
            global test_image
            global box

            new_patch, box = edit_patch_area(test_image, box)
            print(box)

            ax_img.images[0].set_data(new_patch)
            fig.canvas.draw()
        return edit_area

    # Callback for 'Reset' button
    def reset_img_callback(ax_img):
        def reset(event):
            global note_backup
            ax_img.images[0].set_data(note_backup)
            fig.canvas.draw()
        return reset

    # Callback for 'Restore' button
    def restore_img_callback(ax_img, axes_c, index):
        def restore(event):
            global model
            global sde

            # convert image (numpy) to tensor
            image = ax_img.images[0].get_array()
            image = image[:, :, :] / 255.0
            image_torch = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float().unsqueeze(0)
            image_torch = F.interpolate(image_torch, size=(100, 33), mode='bilinear',
                                                 align_corners=False).to(device)
            noisy_state = sde.noise_state(image_torch)

            # convert context images (numpy) to tensors
            c_imgs = []
            for ax in axes_c:
                context = ax.images[0].get_array()
                context = context[:, :, :] / 255.0
                context_torch = torch.from_numpy(np.ascontiguousarray(np.transpose(context, (2, 0, 1)))).float().unsqueeze(0)
                context_resized = F.interpolate(context_torch, size=noisy_state.shape[2:], mode='bilinear',
                                                 align_corners=False).to(device)
                c_imgs.append(context_resized)
            while len(c_imgs) < 3:
                c_imgs.append(c_imgs[0])

            # Feed to the diffusion model to do the restoration
            model.feed_data(torch.cat([noisy_state.to(device), c_imgs[0], c_imgs[1], c_imgs[2]], dim=1), image_torch, image_torch)
            model.test(sde, save_states=True)

            # Return the results
            visuals = model.get_current_visuals()
            SR_img = visuals["Output"]
            output = util.tensor2img(SR_img.squeeze())  # uint8
            ax_img.images[0].set_data(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            fig.canvas.draw()
        return restore


    # Show context notes and buttons on GUI
    for i in range(3):
        x0 = 0.05 + (i + 1) * (0.8 / total_images)

        # Image axis
        ax_img = fig.add_axes([x0, 0.4, 0.8 / total_images, 0.5])
        if i < len(clean_notes):
            ax_img.imshow(clean_notes[i], cmap='gray')
        else:
            ax_img.imshow(clean_notes[0], cmap='gray')
        ax_img.set_title(f'Context {i+1}')
        ax_img.axis('off')
        axes.append(ax_img)

        # Button axis
        ax_btn = fig.add_axes([x0+0.01, 0.29, 0.8 / total_images - 0.02, 0.06])
        button = Button(ax_btn, 'Flip')
        button.on_clicked(make_flip_callback(ax_img))
        buttons.append(button)

        ax_btn_chng = fig.add_axes([x0+0.01, 0.21, 0.8 / total_images - 0.02, 0.06])
        button_chng = Button(ax_btn_chng, 'Change')
        button_chng.on_clicked(make_change_callback(ax_img))
        buttons.append(button_chng)

    # Show degraded note
    ax_main = fig.add_axes([0.05, 0.4, 0.8 / total_images, 0.5])
    ax_main.imshow(note_img, cmap='gray')
    ax_main.set_title('Degraded note')
    ax_main.axis('off')

    ax_btn_edit = fig.add_axes([0.051, 0.29, 0.8/total_images - 0.02, 0.06])
    button_edit = Button(ax_btn_edit, 'Paint')
    button_edit.on_clicked(make_paint_callback(ax_main))
    buttons.append(button_edit)

    ax_btn_edit = fig.add_axes([0.051, 0.21, 0.8/total_images - 0.02, 0.06])
    button_edit = Button(ax_btn_edit, 'Edit area')
    button_edit.on_clicked(make_edit_area_callback(ax_main))
    buttons.append(button_edit)

    ax_btn_reset = fig.add_axes([0.051, 0.13, 0.8/total_images - 0.02, 0.06])
    btn_reset = Button(ax_btn_reset, 'Reset')
    btn_reset.on_clicked(reset_img_callback(ax_main))
    buttons.append(btn_reset)

    ax_btn_restore = fig.add_axes([0.051, 0.05, 0.8/total_images - 0.02, 0.06])
    btn_restore = Button(ax_btn_restore, 'Restore')
    btn_restore.on_clicked(restore_img_callback(ax_main, axes, index))
    buttons.append(btn_restore)

    ax_textbox = fig.add_axes([x0+0.01, 0.05, 0.8/total_images - 0.02, 0.08])
    textbox = TextBox(ax_textbox, 'Note:', initial=label)

    # Machinery for the textbox
    def submit(text):
        global axes
        text = text.strip()
        if text in color_dictionary:
            ax_textbox.set_title(f"Note: {text}")
            clean_notes = find_good_notes(parsed_folium_dict, image_path, text)
            for i, ax in enumerate(axes):
                ax.images[0].set_data(clean_notes[i])
                fig.canvas.draw_idle()
        else:
            ax_textbox.set_title(f"'{text}' not found")
            fig.canvas.draw_idle()


    textbox.on_submit(submit)
    plt.show()
    add_patch(test_image, box, ax_main.images[0].get_array())

plt.imshow(test_image)
plt.title("Restored Image")
plt.axis("off")
plt.show()

plt.imsave("result.png", test_image)

