import os
from functools import cached_property

import pandas as pd
import numpy as np
from scipy.stats import zscore
from skimage.registration import phase_cross_correlation

import config

def calculate_drift(img1, img2):
    q1 = phase_cross_correlation(img1[:1024, :1024], img2[:1024, :1024])[0]
    q2 = phase_cross_correlation(img1[1024:, :1024], img2[1024:, :1024])[0]
    q3 = phase_cross_correlation(img1[1024:, 1024:], img2[1024:, 1024:])[0]
    q4 = phase_cross_correlation(img1[:1024, 1024:], img2[:1024, 1024:])[0]
    df = pd.DataFrame([q1, q2, q3, q4])
    if (df.std() > 15).any():
        df = df[(np.abs(zscore(df)) < 1.25).all(axis=1)]
    return df.median()


def fov_to_global_coordinates(x, y, fov, positions):
    global_x = 220 * x / 2048 + np.array(positions.loc[fov]["y"])
    global_y = 220 * y / 2048 - np.array(positions.loc[fov]["x"])
    return global_x, global_y


def reference_gene_counts(filename: str) -> dict:
    refcounts = pd.read_csv(filename)
    refcounts = dict(zip(refcounts["geneName"], np.log10(refcounts["counts"])))
    return refcounts
