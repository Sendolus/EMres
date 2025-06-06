"""
This file contains a bunch of
"""
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os


def optical_density(s, b):
    """
    Compute the optical density: D(t) = -log(s(t) / b).
    """
    s_clipped = np.clip(s, 1, None)
    return -np.log(s_clipped / b)


def optical_density_to_intensity(D, b):
    """
    Compute the image intensity based on the optimal density.
    """
    return np.clip(b * np.exp(-D), 0, 255)


def compute_observed_density(D, q, h, S, b):
    """
    Compute the observed optical density: D_obs = D - q * log((h ⊗ S) / b).
    """
    observed_density = np.zeros_like(D)
    for c in range(S.shape[2]):  # Iterate over the color channels
        convolved = cv2.filter2D(S[:, :, c], -1, h)
        convolved_clipped = np.clip(convolved, 1, None)  # Avoid division by zero or log of zero
        log_term = np.log(convolved_clipped / b[c])
        observed_density[:, :, c] = D[:, :, c] - q * log_term

    return observed_density


def simulate_bleedtrough(image1, image2, PSFkernel, severity):
    """
    Simulate ink bleedthrough on a recto and a verso page.
    """

    # Resize the second image to equal the dimensions of the first image and flip it
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    image2 = cv2.flip(image2, 1)

    # Simulate ink bleedthrough
    b_r = np.mean(image1, axis=(0, 1))
    b_v = np.mean(image2, axis=(0, 1))

    D_r = optical_density(image1, b_r)
    D_v = optical_density(image2, b_v)

    D_obs_r = compute_observed_density(D_r, severity, PSFkernel, image2, b_v)
    D_obs_v = compute_observed_density(D_v, severity, PSFkernel, image1, b_r)

    image1_degraded = optical_density_to_intensity(D_obs_r, b_r).astype(np.uint8)
    image2_degraded = cv2.flip(optical_density_to_intensity(D_obs_v, b_v).astype(np.uint8), 1)

    return image1_degraded, image2_degraded
