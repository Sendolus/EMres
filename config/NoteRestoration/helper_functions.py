"""
Helper functions that are used for the Graphical User Interface
"""
import cv2
import sys
import numpy as np
import glob
import math
import statistics
import os
import torch
import time

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import collections as mc
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib
from get_label_dict import get_dict_of_label_ids
from skimage.exposure import match_histograms

sys.path.insert(0, "../../")
import utils as util


label_dict = get_dict_of_label_ids()
inv_label_dict = {v: k for k, v in label_dict.items()}

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


def click_damaged_notes(parsed_folium, img_path):
    """
    Indicate notes that are damaged through a popup window by clicking them
    """
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = plt.figure(None, (10, 10), 200)
    fig.add_subplot(111)
    axes = fig.axes[0]
    axes.imshow(image, cmap="gray")

    notes = []
    damaged_notes = set()

    # Define the note bounding boxes from the OMR output
    for index, staff_dict in enumerate(parsed_folium):
        boxes = staff_dict['boxes']
        labels = [inv_label_dict[label_id] for label_id in staff_dict['label_ids']]

        for i in range(0, len(labels)):
            obj = boxes[i]
            label = labels[i]
            x, y, w, h = obj[1], obj[0], obj[3] - obj[1], obj[2] - obj[0]
            if '_' in labels[i]:
                label = label.split('_')[-1]
            color_label = color_dictionary[label]

            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color_label, fill=False)
            axes.add_patch(rect)
            notes.append((rect, f"{index}:{i}", label))

    # Mechanism for clicking or unclicking notes
    def on_click(event):
        for rect, note_nb, label in notes:
            contains, _ = rect.contains(event)
            if contains:
                print(f"Clicked on {note_nb}")
                if (note_nb, label) not in damaged_notes:
                    damaged_notes.add((note_nb, label))
                    rect.set_linewidth(3)
                    rect.set_edgecolor('red')
                else:
                    damaged_notes.remove((note_nb, label))
                    rect.set_linewidth(1)
                    color_label = color_dictionary[label]
                    rect.set_edgecolor(color_label)
                fig.canvas.draw()  # Update the plot
                break

    def on_key(event):
        if event.key == 'enter':
            print("EXIT")
            plt.close(fig)

    # Connect click event to handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.title("Click notes to select, press Enter to finish")
    plt.axis('off')
    plt.show()

    return damaged_notes


def click_one_note(parsed_folium, img_path, on_note_selected):
    """
    This function is connected to the 'change note' button where one new context note can be chosen
    """
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")
    plt.title("Click note to use as new context example")
    plt.axis('off')

    notes = []
    for index, staff_dict in enumerate(parsed_folium):
        boxes = staff_dict['boxes']
        labels = [inv_label_dict[label_id] for label_id in staff_dict['label_ids']]

        for i, label in enumerate(labels):
            obj = boxes[i]
            x, y, w, h = obj[1], obj[0], obj[3] - obj[1], obj[2] - obj[0]
            if '_' in label:
                label = label.split('_')[-1]
            color_label = color_dictionary[label]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color_label, fill=False)
            ax.add_patch(rect)
            notes.append((rect, f"{index}:{i}", label))

    def on_click(event):
        for rect, note_nb, label in notes:
            contains, _ = rect.contains(event)
            if contains:
                print(f"Clicked on {note_nb}")
                plt.close(fig)
                on_note_selected(note_nb, label)
                return

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show(block=False)


def find_good_notes(parsed_folium, img_path, note_type):
    """
    Extract three notes images of the note type 'note_type' with the highest classification scores from the page image.
    """
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    matching = []
    for index, staff_dict in enumerate(parsed_folium):
        boxes = staff_dict['boxes']
        labels = [inv_label_dict[label_id] for label_id in staff_dict['label_ids']]
        scores = staff_dict['scores']

        # Find all notes of type 'note_type'
        for i in range(len(labels)):
            if labels[i] == note_type:
                matching.append((scores[i], boxes[i]))

    matching.sort(reverse=True, key=lambda x: x[0])
    best_3 = [box for _, box in matching[:3]]

    images = []
    for box in best_3:
        y_min, x_min, y_max, x_max = box
        images.append(image[y_min:y_max, x_min:x_max])

    return images


def note_id2image(parsed_folium, img_path, note_id):
    """
    Given a 'note_id' and a page image, extract a crop image of the note in question
    """
    staff, pos = note_id.split(":")
    box = parsed_folium[int(staff)]['boxes'][int(pos)]
    y_min, x_min, y_max, x_max = box

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image[y_min:y_max, x_min:x_max]


def add_patch(image, box, patch):
    """
    Stitch (restored) note patch back into the original image
    """
    y_min, x_min, y_max, x_max = box
    patch = cv2.resize(patch, (x_max-x_min, y_max-y_min))
    image[y_min:y_max, x_min:x_max] = patch


def paint_on_img(image):
    """
    Virtual paintbrush to indicate damaged areas of notes
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    painting = False
    scale = 7
    brush_size = 5

    def draw_circle(event, x, y, flags, param):
        nonlocal painting
        if event == cv2.EVENT_LBUTTONDOWN:
            painting = True
            cv2.circle(image, (x//scale, y//scale), brush_size, (255, 255, 255), -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if painting:
                cv2.circle(image, (x//scale, y//scale), brush_size, (255, 255, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            painting = False
            cv2.circle(image, (x//scale, y//scale), brush_size, (255, 255, 255), -1)

    cv2.namedWindow('Draw, Enter to finish')
    cv2.moveWindow('Draw, Enter to finish', 300, 200)
    cv2.setMouseCallback('Draw, Enter to finish', draw_circle)

    while True:
        disp = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Draw, Enter to finish', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ESC to exit
            break

    cv2.destroyAllWindows()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def edit_patch_area(image, box):
    """
    Edit the bounding box of a note image on the page by 'drawing' a new rectangle with the mouse
    """
    y_min, x_min, y_max, x_max = box

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[y_min-100:y_max+100, x_min-100:x_max+100]
    image_copy = image.copy()
    start_point = None
    painting = False
    end_point = None
    x_mouse, y_mouse = 0, 0
    scale = 2

    def draw_rectangle(event, x, y, flags, param):
        nonlocal painting
        nonlocal start_point
        nonlocal end_point
        nonlocal x_mouse, y_mouse

        if event == cv2.EVENT_LBUTTONDOWN:
            painting = True
            if start_point is None:
                start_point = (x//scale, y//scale)

        elif event == cv2.EVENT_MOUSEMOVE:
            x_mouse, y_mouse = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            painting = False
            end_point = (x//scale, y//scale)
            cv2.rectangle(image, start_point, (x//scale, y//scale), (0, 0, 255), 2)

    cv2.namedWindow('Draw, Enter to finish')
    cv2.moveWindow('Draw, Enter to finish', 300, 200)
    cv2.setMouseCallback('Draw, Enter to finish', draw_rectangle)

    while True:
        temp_image = image.copy()
        if painting and start_point is not None:
            current_point = (x_mouse // scale, y_mouse // scale)
            cv2.rectangle(temp_image, start_point, current_point, (0, 0, 255), 2)
        disp = cv2.resize(temp_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Draw, Enter to finish', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ESC to exit
            break

    cv2.destroyAllWindows()

    min_x = min(start_point[0], end_point[0])
    max_x = max(start_point[0], end_point[0])
    min_y = min(start_point[1], end_point[1])
    max_y = max(start_point[1], end_point[1])
    box = y_min - 100 + min_y, x_min - 100 + min_x, y_min - 100 + max_y, x_min - 100 + max_x

    return cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)[min_y:max_y, min_x:max_x], box
